//! GRID-specific dataset normalization on disk.
//!
//! This module is the **GRID** adapter for forcing raw corpus files into **sharded
//! video–transcript sample bundles**: `grid-lr-corpus/<speaker>/<utterance_id>/` with paired media
//! and transcript files per utterance sample. It discovers legacy or scattered layouts recursively,
//! validates speaker alignment mapping, and **bundles** utterance samples via moves into that layout.
//!
//! Also provides **standard-format conversion** (`.mpg` → `.mp4` via `ffmpeg`, `.align` → `.txt`)
//! and **`clean_corpus`** to remove redundant legacy files after conversion. Other dataset sources
//! should provide their own adapter modules with the same bundle shape, not GRID-specific logic here.



use crate::{
    context::Context,
    prelude::{io_err, ESS},
    pipeline::io::file_nonempty,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::{HashMap, HashSet},
    fs::{
        create_dir_all,
        read_dir,
        remove_file,
        rename,
        File,
    },
    io::{
        BufRead,
        BufReader,
        ErrorKind,
        Write,
    },
    path::{Path, PathBuf},
    process::Command,
};



#[derive(Debug)]
struct GridDiscovery {
    video_paths_by_speaker: HashMap<String, HashMap<String, PathBuf>>,
    align_paths_by_speaker: HashMap<String, HashMap<String, PathBuf>>,
    video_stems_by_speaker: HashMap<String, HashSet<String>>,
    align_stems_by_speaker: HashMap<String, HashSet<String>>,
}



fn is_grid_speaker_dir(name: &str) -> bool {
    if !name.starts_with('s') || name.len() < 2 { return false; }
    name[1..].parse::<i32>().is_ok()
}



fn is_bundled_path(
    grid_root: &PathBuf,
    path: &PathBuf,
    speaker: &str,
    utterance_id: &str,
) -> bool {
    let expected_dir = grid_root.join(speaker).join(utterance_id);
    match path.parent() {
        Some(parent) => parent == expected_dir,
        None => false,
    }
}



fn insert_indexed_path(
    map: &mut HashMap<String, HashMap<String, PathBuf>>,
    grid_root: &PathBuf,
    speaker: &str,
    utterance_id: &str,
    path: PathBuf,
) {
    let entry = map.entry(speaker.to_string()).or_default();
    if let Some(existing) = entry.get(utterance_id) {
        // prefer already-bundled destination paths to avoid picking a legacy duplicate
        let existing_bundled = is_bundled_path(grid_root, &existing.clone(), speaker, utterance_id);
        let new_bundled = is_bundled_path(grid_root, &path, speaker, utterance_id);

        if existing_bundled { return; }
        if new_bundled { entry.insert(utterance_id.to_string(), path); }
    } else { entry.insert(utterance_id.to_string(), path); } // else keep first seen to remain deterministic-ish
}



fn discover_grid_files_at_any_depth(grid_root: &PathBuf) -> GridDiscovery {
    fn walk_dir(
        grid_root: &PathBuf,
        dir: &PathBuf,
        video_paths_by_speaker: &mut HashMap<String, HashMap<String, PathBuf>>,
        align_paths_by_speaker: &mut HashMap<String, HashMap<String, PathBuf>>,
    ) {
        let entries = match read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk_dir(grid_root, &path, video_paths_by_speaker, align_paths_by_speaker);
                continue;
            }

            let ext = match path.extension().and_then(|e| e.to_str()) {
                Some(e) => e,
                None => continue,
            };
            if ext != "mpg" && ext != "align" { continue; }

            let utterance_id = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };

            let speaker = match path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
            {
                Some(s) if is_grid_speaker_dir(s) => s.to_string(),
                _ => continue,
            };

            if ext == "mpg" {
                insert_indexed_path(
                    video_paths_by_speaker,
                    grid_root,
                    &speaker,
                    &utterance_id,
                    path,
                );
            } else {
                insert_indexed_path(
                    align_paths_by_speaker,
                    grid_root,
                    &speaker,
                    &utterance_id,
                    path,
                );
            }
        }
    }

    let mut video_paths_by_speaker: HashMap<String, HashMap<String, PathBuf>> = HashMap::new();
    let mut align_paths_by_speaker: HashMap<String, HashMap<String, PathBuf>> = HashMap::new();
    walk_dir(
        grid_root,
        grid_root,
        &mut video_paths_by_speaker,
        &mut align_paths_by_speaker,
    );

    let mut video_stems_by_speaker: HashMap<String, HashSet<String>> = HashMap::new();
    for (speaker, paths) in &video_paths_by_speaker {
        let stems: HashSet<String> = paths.keys().cloned().collect();
        video_stems_by_speaker.insert(speaker.clone(), stems);
    }

    let mut align_stems_by_speaker: HashMap<String, HashSet<String>> = HashMap::new();
    for (speaker, paths) in &align_paths_by_speaker {
        let stems: HashSet<String> = paths.keys().cloned().collect();
        align_stems_by_speaker.insert(speaker.clone(), stems);
    }

    GridDiscovery {
        video_paths_by_speaker,
        align_paths_by_speaker,
        video_stems_by_speaker,
        align_stems_by_speaker,
    }
}



fn determine_speaker_mapping_from_stems(
    video_stems_by_speaker: &HashMap<String, HashSet<String>>,
    align_stems_by_speaker: &HashMap<String, HashSet<String>>,
) -> HashMap<String, String> {
    let mut map = HashMap::new();

    for (video_speaker, video_stems) in video_stems_by_speaker {
        let best = align_stems_by_speaker
            .iter()
            .map(|(align_speaker, align_stems)| {
                let overlap = video_stems.intersection(align_stems).count();
                let same_name = align_speaker == video_speaker;
                (overlap, same_name, align_speaker.clone())
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
            .map(|(_, _, sp)| sp)
            .unwrap_or_else(|| video_speaker.clone());

        map.insert(video_speaker.clone(), best);
    }

    map
}



fn assert_speaker_mapping_is_not_many_to_one(mapping: &HashMap<String, String>) -> Result<(), ESS> {
    let mut align_to_video: HashMap<String, Vec<String>> = HashMap::new();
    for (v, a) in mapping
    { align_to_video.entry(a.clone()).or_default().push(v.clone()); }

    for (a, vs) in &align_to_video {
        if vs.len() > 1
        { return Err(io_err(format!("Ambiguous mapping: alignment speaker {} is best match for multiple video speakers: {:?}", a, vs), ErrorKind::InvalidInput)); }
    }

    Ok(())
}



/// Visits each `grid_root/<speaker>/<utterance_id>/` directory where `speaker` matches GRID layout and returns a list of (utterance_path, speaker, utterance_id) items.
fn list_bundled_dirs(grid_root: &Path) -> Result<Vec<(PathBuf, String, String)>, ESS> {
    let mut bundles_list = Vec::new();
    let rd = read_dir(grid_root)
        .map_err(|e| { io_err(format!("failed to read GRID corpus dir {:?}: {}", grid_root, e), ErrorKind::Other) })?;

    for speaker_ent in rd.flatten() {
        let speaker_path = speaker_ent.path();
        if !speaker_path.is_dir() { continue; }

        let speaker = speaker_ent
            .file_name()
            .to_str()
            .unwrap_or("")
            .to_string();

        if !is_grid_speaker_dir(&speaker) { continue; }
        let entries = read_dir(&speaker_path)
            .map_err(|e| { io_err(format!("failed to read speaker dir {:?}: {}", speaker_path, e), ErrorKind::Other) })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() { continue; }

            let entry_id = entry
                .file_name()
                .to_str()
                .unwrap_or("")
                .to_string();
            if entry_id.is_empty() { continue; }

            bundles_list.push((path, speaker.clone(), entry_id));
        }
    }

    Ok(bundles_list)
}



/// Converts a GRID MPEG-1 source video to H.264 MP4 in the same utterance directory.
///
/// ### Params:
/// - `src_mpg`: Path to existing `.../<utterance_id>/<utterance_id>.mpg`.
/// - `dest_mp4`: Path to write `.../<utterance_id>/<utterance_id>.mp4`.
///
/// ### Returns:
/// `Ok(())` on success; error if `src_mpg` missing, `ffmpeg` fails, or output not created.
///
/// Idempotent: if `dest_mp4` exists and is non-empty, returns `Ok(())` immediately.
pub fn convert_to_standard_mp4(src_mpg: &Path, dest_mp4: &Path) -> Result<(), ESS> {
    if file_nonempty(dest_mp4) { return Ok(()); }
    if !src_mpg.is_file() { return Err(io_err(format!("source mpg not found: {:?}", src_mpg), ErrorKind::NotFound)); }

    if let Some(parent) = dest_mp4.parent() {
        create_dir_all(parent)
            .map_err(|e| { io_err(format!("failed to create parent dir {:?}: {}", parent, e), ErrorKind::Other) })?;
    }

    let src_str = src_mpg.to_str()
        .ok_or_else(|| { io_err("source mpg path is not valid UTF-8", ErrorKind::InvalidInput) })?;

    let dest_str = dest_mp4.to_str()
        .ok_or_else(|| { io_err("destination mp4 path is not valid UTF-8", ErrorKind::InvalidInput) })?;

    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            src_str,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            dest_str,
        ])
        .output()
        .map_err(|e| {
            if e.kind() == ErrorKind::NotFound
            { io_err("ffmpeg not found on PATH; install ffmpeg to convert .mpg to .mp4", ErrorKind::NotFound) }
            else { io_err(format!("failed to spawn ffmpeg: {}", e), ErrorKind::Other) }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io_err(format!("ffmpeg failed: {}", stderr.trim()), ErrorKind::Other));
    }

    if !file_nonempty(dest_mp4)
    { return Err(io_err(format!("ffmpeg produced no output at {:?}", dest_mp4), ErrorKind::Other)); }

    Ok(())
}



/// Parses GRID `.align` lines into space-separated reference words (skips `sil` / `sp`).
pub fn parse_align(src_align: &Path) -> Result<String, ESS> {
    let file = File::open(src_align)
        .map_err(|e| { io_err(format!("failed to open align file {:?}: {}", src_align, e), ErrorKind::Other) })?;

    let mut words_out: Vec<String> = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line_group = line.split_whitespace().collect::<Vec<_>>();
        if line_group.len() < 3 { continue; }

        let word = line_group[2];
        if word != "sil" && word != "sp" { words_out.push(word.to_string()); }
    }

    if words_out.is_empty()
    { return Err(io_err(format!("no non-silence words found in {:?}", src_align), ErrorKind::InvalidData)); }

    Ok(words_out.join(" "))
}



/// Writes a reference transcript from a GRID `.align` file to UTF-8 `.txt` (word-level, space-separated).
///
/// ### Params:
/// - `src_align`: Path to `.../<utterance_id>/<utterance_id>.align`.
/// - `dest_txt`: Path to write `.../<utterance_id>/<utterance_id>.txt`.
///
/// ### Returns:
/// `Ok(())` on success.
///
/// Idempotent: if `dest_txt` exists and is non-empty, returns `Ok(())` immediately.
pub fn convert_to_standard_txt(src_align: &Path, dest_txt: &Path) -> Result<(), ESS> {
    if file_nonempty(dest_txt) { return Ok(()); }
    if !src_align.is_file()
    { return Err(io_err(format!("source align not found: {:?}", src_align), ErrorKind::NotFound)); }

    let line = parse_align(src_align)?;
    let tmp_path = dest_txt.with_extension("tmp");
    {
        let mut f = File::create(&tmp_path)
            .map_err(|e| { io_err(format!("failed to create temp file {:?}: {}", tmp_path, e), ErrorKind::Other) })?;
        f.write_all(line.as_bytes())
            .map_err(|e| { io_err(format!("failed to write temp file {:?}: {}", tmp_path, e), ErrorKind::Other) })?;
    }

    rename(&tmp_path, dest_txt).map_err(|e| {
        let _ = remove_file(&tmp_path);
        io_err(format!("failed to rename {:?} -> {:?}: {}", tmp_path, dest_txt, e), ErrorKind::Other)
    })?;

    Ok(())
}



/// Runs `convert_to_standard_mp4` and `convert_to_standard_txt` for every bundled utterance under the GRID corpus.
///
/// ### Params:
/// - `context`: Filesystem context (`data/grid-lr-corpus`).
///
/// ### Returns:
/// `Ok(())` when the pass completes; errors if any conversion fails.
pub fn normalize_to_standard_formats(context: &Context) -> Result<(), ESS> {
    let grid_path = context.data_path.join("grid-lr-corpus");
    if !grid_path.is_dir()
    { return Err(io_err(format!("GRID corpus directory does not exist at {:?}", grid_path), ErrorKind::NotFound)); }

    let bundles = list_bundled_dirs(&grid_path)?;
    let n = bundles.len() as u64;

    println!("Standardizing GRID corpus for {} samples...", n);

    let prog_bar = ProgressBar::new(n);
    prog_bar.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) (ETA: {eta})\n")
        .unwrap()
        .progress_chars("#>-"),
    );

    for (dir, speaker, id) in bundles {
        prog_bar.set_message(format!("{}/{}", speaker, id));

        let mpg = dir.join(format!("{}.mpg", id));
        let mp4 = dir.join(format!("{}.mp4", id));
        if mpg.is_file() { convert_to_standard_mp4(&mpg, &mp4)?; }

        let align = dir.join(format!("{}.align", id));
        let txt = dir.join(format!("{}.txt", id));
        if align.is_file() { convert_to_standard_txt(&align, &txt)?; }

        prog_bar.inc(1);
    }

    prog_bar.finish_with_message("GRID standard-format pass finished (.mp4 / .txt, idempotent per file)");
    println!("\n");

    Ok(())
}



/// Removes GRID corpus files redundant after standardization (e.g. `.mpg` when `.mp4` exists).
///
/// ### Params:
/// - `context`: Filesystem context (`data/grid-lr-corpus` root derived from `context.data_path`).
/// - `dry_run`: If true, only print paths that would be removed.
///
/// ### Returns:
/// `Ok(())` on success, or an error if deletion fails.
///
/// ### Safety:
/// Only deletes when the replacement standard file exists and is non-empty.
pub fn clean_corpus(context: &Context, dry_run: bool) -> Result<(), ESS> {
    let grid_path = context.data_path.join("grid-lr-corpus");
    if !grid_path.is_dir()
    { return Err(io_err(format!("GRID corpus directory does not exist at {:?}", grid_path), ErrorKind::NotFound)); }

    let bundles = list_bundled_dirs(&grid_path)?;
    let mut removed = 0usize;

    for (dir, _, id) in bundles {
        let mpg = dir.join(format!("{}.mpg", id));
        let mp4 = dir.join(format!("{}.mp4", id));
        if file_nonempty(&mp4) && mpg.is_file() {
            if dry_run { println!("[dry-run] would remove {:?}\n", mpg); }
            else {
                remove_file(&mpg)
                    .map_err(|e| { io_err(format!("failed to remove {:?}: {}", mpg, e), ErrorKind::Other) })?;
            }
            removed += 1;
        }

        let align = dir.join(format!("{}.align", id));
        let txt = dir.join(format!("{}.txt", id));
        if file_nonempty(&txt) && align.is_file() {
            if dry_run { println!("[dry-run] would remove {:?}\n", align); }
            else {
                remove_file(&align)
                    .map_err(|e| { io_err(format!("failed to remove {:?}: {}", align, e), ErrorKind::Other) })?;
            }
            removed += 1;
        }
    };

    if dry_run { println!("Clean corpus dry-run: {} file(s) would be removed\n", removed); }
    else { println!("Clean corpus: removed {} redundant file(s)\n", removed); }

    Ok(())
}



/// Validates and reports the best-match mapping from video speakers to alignment speakers.
///
/// Discovery is recursive under `data/grid-lr-corpus` and does not assume any particular
/// folder structure; the speaker ID is inferred from the immediate parent directory name
/// (e.g. `s1`, `s2`, ...).
///
/// This function is validate-only: it does not rename any directories. The mapping is
/// intended to be applied by `bundle_grid_utterances()` when normalizing the corpus.
pub fn align_grid_directories(context: &Context, dry_run: bool) -> Result<(), ESS> {
    let grid_path = context.data_path.join("grid-lr-corpus");

    if !grid_path.exists()
    { return Err(io_err(format!("GRID corpus directory does not exist at {:?}", grid_path), ErrorKind::NotFound)); }

    let discovery = discover_grid_files_at_any_depth(&grid_path);
    let mapping = determine_speaker_mapping_from_stems(
        &discovery.video_stems_by_speaker,
        &discovery.align_stems_by_speaker,
    );
    assert_speaker_mapping_is_not_many_to_one(&mapping)?;

    if dry_run {
        println!("[DRY RUN] Speaker mapping (video -> alignment):");
        let non_identity: Vec<_> = mapping.iter().filter(|(k, v)| k != v).collect();
        if non_identity.is_empty() { println!("  No non-identity mappings; all alignments already match videos\n"); }
        else { for (v, a) in non_identity { println!("  {} -> {}", v, a); } }

        return Ok(());
    }

    let non_identity: Vec<_> = mapping.iter().filter(|(k, v)| k != v).collect();
    if !non_identity.is_empty() { println!( "Discovered speaker mapping (video -> alignment, non-identity): {:?}\n", non_identity); }

    Ok(())
}



/// Bundles GRID utterance ssamples into the normalized bundled layout:
/// `grid-lr-corpus/<speaker>/<utterance_id>/<utterance_id>.{mpg,align}`
///
/// Discovery is recursive under `data/grid-lr-corpus` and does not assume any particular
/// folder structure; speaker IDs are inferred from the immediate parent directory name.
///
/// Uses `align_grid_directories()`'s mapping logic (validate-only) to pair video speakers
/// with the best matching alignment speakers.
///
/// Idempotent per utterance sample: if the destination pair already exists, it will be skipped.
pub fn bundle_grid_utterances(context: &Context) -> Result<(), ESS> {
    let grid_path = context.data_path.join("grid-lr-corpus");

    if !grid_path.exists()
    { return Err(io_err(format!("GRID corpus directory does not exist at {:?}", grid_path), ErrorKind::NotFound)); }

    let discovery = discover_grid_files_at_any_depth(&grid_path);
    let mapping = determine_speaker_mapping_from_stems(
        &discovery.video_stems_by_speaker,
        &discovery.align_stems_by_speaker,
    );
    assert_speaker_mapping_is_not_many_to_one(&mapping)?;

    let mut moved = 0usize;
    for (video_speaker, utterances) in &discovery.video_paths_by_speaker {
        for (utterance_id, src_mpg) in utterances {
            let align_speaker = mapping
                .get(video_speaker)
                .map(|s| s.as_str())
                .unwrap_or(video_speaker.as_str());

            let src_align = match discovery
                .align_paths_by_speaker
                .get(align_speaker)
                .and_then(|m| m.get(utterance_id))
            {
                Some(p) => p,
                None => {
                    eprintln!(
                        "Skipping {}/{}: missing alignment for mapped speaker {}",
                        video_speaker, utterance_id, align_speaker
                    );
                    continue;
                }
            };

            let utterance_dir = grid_path.join(video_speaker).join(utterance_id);
            let dest_mpg = utterance_dir.join(format!("{}.mpg", utterance_id));
            let dest_align = utterance_dir.join(format!("{}.align", utterance_id));
            if dest_mpg.exists() && dest_align.exists() { continue; }

            create_dir_all(&utterance_dir)
                .map_err(|e| { io_err(format!("failed to create utterance dir {:?}: {}", utterance_dir, e), ErrorKind::Other) })?;

            if !dest_mpg.exists() && src_mpg != &dest_mpg && src_mpg.exists() {
                rename(src_mpg, &dest_mpg)
                    .map_err(|e| { io_err(format!("failed to move {:?} -> {:?}: {}", src_mpg, dest_mpg, e), ErrorKind::Other) })?;
                moved += 1;
            }

            if !dest_align.exists() && src_align != &dest_align && src_align.exists() {
                rename(src_align, &dest_align)
                    .map_err(|e| { io_err(format!("failed to move {:?} -> {:?}: {}", src_align, dest_align, e), ErrorKind::Other) })?;
                moved += 1;
            }
        }
    }

    println!("Bundled GRID utterances into speaker/utterance dirs (moved {} files)\n", moved);

    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        env,
        process,
        io::Write,
    };

    #[test]
    fn convert_to_standard_txt_writes_words() {
        let dir = env::temp_dir().join(format!("lrm_grid_txt_{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let align = dir.join("utt.align");
        let txt = dir.join("utt.txt");

        let mut f = File::create(&align).unwrap();
        writeln!(f, "0 1000 sil").unwrap();
        writeln!(f, "1000 2000 hello").unwrap();
        writeln!(f, "2000 3000 world").unwrap();
        drop(f);

        convert_to_standard_txt(&align, &txt).unwrap();
        let s = fs::read_to_string(&txt).unwrap();

        assert_eq!(s.trim(), "hello world");
        convert_to_standard_txt(&align, &txt).unwrap();
    }
}
