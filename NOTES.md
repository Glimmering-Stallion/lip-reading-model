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



# Char-level vs. Word-level N-Gram LM

In a CTC (Connectionist Temporal Classification) beam search where the decoder operates on characters but uses a word-level \(n\)-gram language model (LM), a word boundary is a specific character—typically a space (" ") or a dedicated separator symbol (like "|")—that signals the completion of a word.

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