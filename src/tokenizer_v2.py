"""
Coordinate Tokenizer for the Chess Challenge.

This module implements a custom tokenizer that drastically reduces the vocabulary size 
required for the chess model (from ~1700 tokens to ~75 tokens).

STRATEGY: "Coordinate Tokenization"
-----------------------------------
Instead of treating an entire move like "WPe2e4" as a single unique token, 
we strip the redundant piece information and split the move into its 
coordinate components.

Examples:
    "WPe2e4"  -> [ "e2", "e4" ]
    "BNg8f6"  -> [ "g8", "f6" ]
    "BPa7a8q" -> [ "a7", "a8", "q" ] (Promotion)

RATIONALE:
----------
1.  **Parameter Efficiency**: 
    - Old Vocab: ~1700 tokens * 128 dim = ~217,000 parameters.
    - New Vocab: ~75 tokens * 128 dim   = ~9,600 parameters.
    - SAVINGS: ~208,000 parameters (20% of the 1M budget!).
    - This allows us to add ~2 extra Transformer layers.

2.  **No Information Loss**:
    - The piece type ("WP", "BN") is redundant if the model tracks the board state.
    - If the model knows "e2" has a white pawn, "e2e4" implies a pawn move.
    - We force the model to learn board state tracking, which is desirable.

VOCABULARY STRUCTURE:
---------------------
The vocabulary is fixed and small (~75 tokens):
1.  **Special Tokens**: [PAD], [BOS], [EOS], [UNK]
2.  **Squares**: a1, a2, ... h8 (64 tokens)
3.  **Promotion Pieces**: q, r, b, n (4 tokens)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class CoordinateTokenizer(PreTrainedTokenizer):
    """
    A specific tokenizer for Chess that relies on board coordinates.
    
    This tokenizer breaks down moves into their fundamental geometric components:
    start_square and end_square. Use this to save parameter budget.
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # -------------------------------------------------------------------------
    # Special Tokens
    # -------------------------------------------------------------------------
    PAD_TOKEN = "[PAD]"  # Used for padding batches to the same length
    BOS_TOKEN = "[BOS]"  # Beginning Of Sequence (start of game)
    EOS_TOKEN = "[EOS]"  # End Of Sequence (end of game)
    UNK_TOKEN = "[UNK]"  # Unknown token (should rarely be used)
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        Initialize the CoordinateTokenizer.
        
        Args:
            vocab_file: Path to a generic vocab.json (optional).
            vocab: Direct dictionary injection (optional).
            **kwargs: Arguments passed to PreTrainedTokenizer.
        """
        # Set standard special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN
        
        # Clean kwargs to prevent conflicts with parent class
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)
        
        # Initialize the fixed vocabulary
        # Unlike standard NLP tokenizers, we don't need to "train" this vocabulary
        # because the set of chess squares (a1-h8) is fixed and known.
        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._create_fixed_vocab()
            
        # create reverse mapping for decoding (ID -> Token)
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )
        
    def _create_fixed_vocab(self) -> Dict[str, int]:
        """
        Generates the static vocabulary for chess coordinates.
        
        Returns:
            Dict[str, int]: A mapping from token string to integer ID.
        """
        vocab = {}
        idx = 0
        
        # 1. Add Special Tokens first
        for token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            vocab[token] = idx
            idx += 1
            
        # 2. Add all 64 Squares (a1 ... h8)
        # We iterate Files (a-h) then Ranks (1-8)
        files = "abcdefgh"
        ranks = "12345678"
        for r in ranks:
            for f in files:
                square = f + r  # e.g., "a1", "e4", "h8"
                vocab[square] = idx
                idx += 1
                
        # 3. Add Promotion Suffixes
        # When a pawn promotes, we need to know what it becomes:
        # q=Queen, r=Rook, b=Bishop, n=Knight
        for p in ["q", "r", "b", "n"]:
            vocab[p] = idx
            idx += 1
            
        return vocab

    @property
    def vocab_size(self) -> int:
        """Returns the total number of tokens in the vocabulary."""
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary dictionary."""
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of space-separated chess moves into coordinates.
        
        Logic:
           Input: "WPe2e4 NBg8f6"
           1. Split by space: ["WPe2e4", "NBg8f6"]
           2. Clean each move:
              - Remove suffixes like (+), (x)
              - "WPe2e4" -> ends in "e2e4"
              - Split "e2e4" -> ["e2", "e4"]
           Output: ["e2", "e4", "g8", "f6"]
        
        Args:
            text (str): Input string of moves.
            
        Returns:
            List[str]: List of coordinate tokens.
        """
        tokens = []
        moves = text.strip().split()
        
        for move in moves:
            # --- Step A: Cleaning ---
            # Remove annotation suffixes found in Lichess dataset
            # (x) = capture, (+) = check, (#) or (+*) = mate, (o)/(O) = castle (ignored as handled by king move)
            clean_move = move.replace("(x)", "") \
                             .replace("(+)", "") \
                             .replace("(+*)", "") \
                             .replace("(o)", "") \
                             .replace("(O)", "")
            
            # --- Step B: Handling Promotion ---
            # Moves like "a7a8q" (promotion to queen) have 5 relevant chars at the end
            promotion_char = None
            if clean_move[-1] in "qrbn":
                promotion_char = clean_move[-1]
                clean_move = clean_move[:-1]  # Strip the promotion char temporarily
            
            # --- Step C: Extracting Coordinates ---
            # A valid move must now end with 4 characters representing 2 squares (e.g., "e2e4")
            # We ignore the piece prefix ("WP", "BN", "WR") completely.
            if len(clean_move) >= 4:
                # Last 2 chars = To Square
                to_square = clean_move[-2:]
                # 2 chars before that = From Square
                from_square = clean_move[-4:-2]
                
                # Check validity (simple heuristic)
                if from_square in self._vocab and to_square in self._vocab:
                    tokens.append(from_square)
                    tokens.append(to_square)
                    
                    if promotion_char:
                        tokens.append(promotion_char)
                else:
                    # If parsing fails (unexpected format), use UNK
                    tokens.append(self.UNK_TOKEN)
            else:
                # Token too short to be a move (e.g. metadata artifacts?)
                tokens.append(self.UNK_TOKEN)
                
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Map a token string to its integer ID."""
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN))

    def _convert_id_to_token(self, index: int) -> str:
        """Map an integer ID back to its token string."""
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens back into a single string.
        
        Note: This does NOT perfectly reconstruct the original "WPe2e4" format
        because we discarded the piece/color information ("WP"). 
        It reconstructs a minimal "e2e4" format which is valid for evaluation.
        
        Input: ["e2", "e4", "g8", "f6"]
        Output: "e2e4 g8f6" (Space separated moves)
        """
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        
        # We need to buffer tokens to reconstruct moves (pairs of squares).
        # Simple heuristic: Join all tokens, then let the user or evaluator 
        # split them. However, readable output is better.
        
        # We'll just join them with spaces for now, as that's the safest
        # generic implementation. The Evaluator class in src/evaluate.py
        # might need to handle "e2 e4" vs "e2e4".
        # But wait! Standard UCI is "e2e4". "e2 e4" is two valid squares but not a move.
        
        # Let's try to be smart:
        # This function is used mostly for decoding generated text.
        # If we return "e2 e4 g8 f6", we need a post-processor to merge e2+e4.
        
        # Join with empty string to produce standard UCI (e.g. "e2e4")
        filtered_tokens = [t for t in tokens if t not in special]
        return "".join(filtered_tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save the vocabulary to a file (required by Hugging Face).
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
            
        return (vocab_file,)
