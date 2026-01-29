# High-Level Overview

## 1. The Video Brain (The Model)

You feed video frames into your Neural Network. The model doesn't see "words" yet; it sees visual patterns. It outputs Logits (basically a giant grid of "confidence scores" for every character in your VOCAB for every single frame).

## 2. The Training Guide (CTC Loss)

During training, the model might guess "hhhh-ee-lll-o".

The Problem: The video has 100 frames, but the word "hello" only has 5 letters. How does the computer know which frames match which letters?

The Fix: CTC Loss is a math trick that calculates all possible ways "hello" could fit into those 100 frames. It tells the model: 

"Adjust your weights so that some combination of these frames equals 'hello'." You use it as the judge during the training loop to tell the model how wrong it was.

## 3. The Raw Output (Logits)

During evaluation (inference), the model emits those Logits (scores) for a new video it has never seen.

- Frame 1: 90% chance of 'h'
- Frame 15: 85% chance of 'h'
- Frame 40: 70% chance of 'blank'
- Frame 75: 92% chance of 'e'
- ...and so on.

## 4. The Smart Filter (CTC Decoder + N-gram LM)

The CTC Decoder takes those messy scores and cleans them up.

The Decoder's Job: It collapses repeats (turning "hhhh" into "h") and removes "blanks."

The N-gram's Job: It acts as the "English Teacher." If the decoder is torn between "hellow" and "hello," your N-gram LM whispers: "In English, 'hello' is much more common than 'hellow'."

The result: You get a clean, character-level prediction like "hello" from a video of someone talking.



# Why Rust?

- Rust's compile-time checks and direct machine code compilation should allow for much greater determinism in forward passing and data transformations across runs.

- Direct machine code running should also allow for super-low latency model inferencing and zero runtime overhead.

- Rust's ownership/borrow system eliminates entire groups of runtime errors, classically encountered in C++ (like dangling pointers, double-frees, null dereferences, etc.), and a Garbage Collector with Python.

- Unlike Python with its Global Interpreter Lock, Rust's fearlessly concurrent nature allows us to harness multi-core parallelism (in our case, Burn's ```DataLoader``` spawning multi-worker threads to fetch/process frames, or concurrently collating data on CPU before moving to GPU in ```VsrmBatcher```).

- Rust compiles projects into one single, static, lightweight binary which completely avoids container bloat problems (like with Python + CUDA + PyTorch payloads as an example), such as to reduce deployment sizes from the Gb range down to a few Mb.


# Char-level vs. Word-level N-Gram LM

In a CTC (Connectionist Temporal Classification) beam search where the decoder operates on characters but uses a word-level N-gram language model (LM), a word boundary is a specific character—typically a space (" ") or a dedicated separator symbol (like "|")—that signals the completion of a word.

This boundary is critical because it tells the decoder when to query the word-level LM for a score.

1. Triggering the Language Model In this hybrid setup, the LM cannot score "partial" words like "ca" or "cat" if it only knows full words. During a word: As the decoder adds non-boundary characters (e.g., 'c', 'a', 't'), it typically only uses the acoustic model score.At the boundary: When a space is predicted, the decoder identifies the preceding character sequence as a completed word (e.g., "cat"). It then queries the \(n\)-gram LM to get the probability of "cat" appearing after the previous word.

2. Lexicon Constraints To manage the mismatch between characters and words, many decoders use a Lexicon (a prefix tree or trie): The beam search only allows character sequences that form valid prefixes of words in the lexicon.A word boundary can only be legally placed if the current character sequence matches a full word in that lexicon.

3. Look-Ahead Scoring (Advanced) Some sophisticated decoders, like "Word Beam Search," don't wait for the physical space character to apply a score. Instead, they "look ahead" into the prefix tree to estimate the LM probability of all possible words that could be completed from the current partial string, effectively treating every step as a potential path to a word boundary.

4. Special Character Roles It is important to distinguish between three types of "separators" in this context: Space (" "): A standard character in your vocabulary that acts as the linguistic word boundary.CTC Blank ("-"): A technical symbol used by the acoustic model to separate repeated characters or indicate "no character"; it is not a word boundary.Separator ("|"): Often used in specific LM formats (like KenLM) to explicitly mark word ends when the model is trained on character sequences.

## The Core Concept

The "space" is the trigger. You don't just track "space vs. non-space"; you track "complete words" vs. "in-progress words."

- Non-Space Characters (In-Progress): When you extend a prefix with a letter (e.g., t after ca), the word-level LM remains silent (score = 0) because "cat" isn't a finished word yet. You rely solely on the Acoustic Model.

- Space Character (Boundary): The moment your code hits the space character, you extract the characters since the last space (e.g., c-a-t), look up the word "cat" in your LM, and apply the \(n\)-gram penalty/bonus all at once.

Why it’s slightly more complex

To make this work in your per_sample_decode function, you need to handle two things:

- The "History" State: Your LM query lm.next_log_prob needs to know the previous word(s), not just the previous characters. You must track the sequence of words already completed in that beam.

- The Lexicon (Crucial): In a pure word-level LM, if the character sequence before the space (e.g., "xyzpdq") isn't in your vocabulary, the LM score is \(-\infty \). Most decoders use a Lexicon/Trie to prevent the beam from even considering character sequences that don't form valid words.

### Summary for your code:

- If v is a letter: lm_score = 0 (or use a character-level "look-ahead" if you're fancy).

- If v is a space: lm_score = LM.score(word_just_finished | previous_words).

Your current code applies lm.next_log_prob on every character extension. For a word-level LM, you should wrap that logic in an if v == space_id block.

## 1. Code Changes Required

The modifications are moderate but require structural changes to your prefix state and extension logic:

- Modify BeamPrefix Struct: Add a field to track the current_word (characters since the last space) and word_history (list of completed words).

- Implement a Lexicon (Trie): Before extending a prefix with a character v, check if current_word + v exists in your trie. If it doesn't, prune that path immediately. This prevents the model from hallucinating gibberish.

- Trigger LM on Space: Move your language_model.next_log_prob call inside a conditional block:
    - If v is a letter: Score = 0 (Acoustic Model only).
    - If v is a space: Score = LM.score(current_word | word_history). Then, clear current_word and add it to word_history.

- EOS Handling: At the end of the timesteps, apply one final LM score for the last unfinished word in each beam.

## 2. WER Performance Difference

Integrating a word-level LM and lexicon typically provides a significant boost in lip reading performance compared to character-level decoding:

- Word Error Rate (WER): Expect a relative improvement of 30–50%. Lip reading is inherently ambiguous (e.g., "p" and "b" look identical); a word-level LM resolves this by forcing the output to follow valid linguistic patterns.

- Out-of-Vocabulary (OOV) Risk: The primary downside is that a lexicon-based decoder cannot predict words it hasn't seen before. If the speaker says a name not in your dictionary, the model will be forced to pick the closest sounding "valid" word.

- Character Error Rate (CER): CER may stay similar or slightly increase if the LM "corrects" a single letter into a completely different (but linguistically likely) word.

# How CTC Loss Forward Works

Purpose: To calculate the total probability of all valid ways a frame-by-frame prediction sequence can be condensed into a specific text sequence target such as to allow the model to learn without pre-aligned timing info.

1. Sequence Augmentation: It expands the target sequence by interleaving blank tokens (e.g., "CAT" becomes "\_C_A_T_"). With a base target sequence of $L$, this creates a blank-extended target sequence of length $2L+1$, so that distinguishing between intentional and unintentional symbol repeats becomes possible (e.g. "good" vs. "god").

2. Dynamic Programming (DP): It maintains a forward variable buffer that stores the total log-probability of all paths reaching a specific symbol in the augmented target sequence at time $t$.

3. State Transitions: Per frame/timestep and per symbol in the blank-extended target sequence (time-sequence grid), it calculates the probability of being at the current symbol by looking at three possible previous states:
    - Stay: Remaining on the same symbol (accounting for repeated predictions).
    - Advance by 1: Moving from the immediate previous symbol (blank or non-blank) to the current symbol.
    - Advance by 2 (Skip): Moving from two symbols back to the current symbol. This is only allowed if the symbol two positions back was a character and that character is different from the current character (effectively skipping a blank).

4. Log-Domain Stability: It uses LogSumExp (or LSE) to perform additions in the log-probability space, which avoids numerical underflow that occurs when multiplying many small probabilities over long sequences.

5. Final Aggregation: The loss is the negative log-probability of the sum of the last two possible states (the final character or the final blank) at the last frame/timestep $T$.

# How CTC Decode Forward Works

Purpose: To transform frame-by-frame logits made by the model into a final text sequence prediction. There are two common methods for CTC Decoding: Greedy Search and Prefix Beam Search.

## Greedy Search

Greedy decoding uses a best-path approach that assumes the most probable sequence can be found by picking the most likely token at every single frame independently of all other frames.

1. Frame-wise Argmax: Per frame/timestep $T$ and per symbol in the vocabulary, the algorithm looks at the logit distribution and picks the ID with the highest probability. It ignores all other candidates at this stage.

2. Best-Path Construction: It stitches these individual winners together into a single raw sequence (the path) of length $T$.

3. Deduplication: Since the model might predict the same character over multiple frames (e.g., AAA), a path collapse function merges consecutive identical tokens into one (e.g., A).

4. Blank Removal: Finally, it strips out the blank tokens. Because blanks were inserted between repeats during training, this step makes sure that a sequence like A_AA (where _ is blank) collapses to AA, while AAA collapses to A.

## Prefix Beam Search

Prefix Beam decoding manages a collection of prefixes and sums the probabilities of all paths that could produce them.

1. Dual-State Tracking: Instead of tracking raw paths, it maintains a beam of collapsed prefixes. For each prefix, it tracks two separate log-probabilities: one where the path ends in a blank and one where it ends in a non-blank character, which is the core idea here for handling character repetitions correctly.

2. Successive Expansion: Per frame/timestep and per symbol in the vocabulary (time-vocabulary grid), the algorithm attempts to extend every prefix in the beam with every possible character in that vocabulary, while accounting for three possible cases:
    - Blank Extension: Moves the total probability of the prefix into the "ending in blank" state.
    - Repeated Character: If the new character matches the prefix's last character, it only extends the sequence if the previous path ended in a blank (Case B). If it ended in a non-blank, it is treated as a "stretch" of the existing character (Case A).
    - New Character: If the character is different, it appends it to the prefix regardless of the previous state (Case C).

3. Path Merging (Consolidation): It uses a hash map to group different paths that collapse into the same text prefix. By summing these probabilities (via log_sum_exp), it finds the total probability of a label rather than just a single path.

4. Language Model (LM) Fusion: During expansion, the algorithm integrates an external Language Model. It adds a "reward" to prefixes that are linguistically likely, helping the decoder choose correct words when the acoustic model is uncertain.

5. Pruning and Scoring: To keep the search efficient, the algorithm sorts the prefixes by a combined score—incorporating acoustic probability, LM score, and a length bonus—and keeps only the top \(N\) candidates (the beam_width) for the next frame/timestep.

6. Final Selection: After processing all frames, it selects the prefix with the highest overall score as the final prediction. 