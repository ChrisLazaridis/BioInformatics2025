import math
import numpy as np
import random
from Bio import SeqIO
from typing import List, Tuple

# Define the 5-symbol alphabet explicitly
ALPHABET = ['A', 'C', 'G', 'T', '_']


def entropy(column: List[str]) -> float:
    """
    Shannon entropy of a column over the alphabet {A,C,G,T,_}.
    """
    N = len(column)
    H = 0.0
    for sym in ALPHABET:
        count = column.count(sym)
        if count == 0:
            continue
        p = count / N
        H -= p * math.log2(p)
    return H


def information_content(column: List[str]) -> float:
    """
    Information content IC(c) = log2(|ALPHABET|) - H(c)
    """
    return math.log2(len(ALPHABET)) - entropy(column)


class HMMProfile:
    """
    Profile HMM built from an MSA. IC threshold → match states.
    Viterbi decoding + Viterbi-training with fine-tuning and Dirichlet prior using Π.
    """
    def __init__(self, aligned_sequences: List[str], ic_threshold: float = 1.0, pseudocount: float = 1e-6):
        self.aligned_sequences = aligned_sequences
        self.N = len(aligned_sequences)
        self.L = len(aligned_sequences[0])
        self.ic_threshold = ic_threshold
        self.pseudocount = pseudocount

        # Background symbol distribution Π from MSA (exclude gaps)
        counts = {sym: 0 for sym in ALPHABET if sym != '_'}
        for seq in aligned_sequences:
            for char in seq:
                if char in counts:
                    counts[char] += 1
        total_counts = sum(counts.values())
        self.symbol_pi = {sym: counts[sym] / total_counts for sym in counts}

        # Define strong (match) columns by IC
        cols = list(zip(*aligned_sequences))
        self.strong_cols = [information_content(list(col)) > ic_threshold for col in cols]
        self.match_col_idx = [i for i, strong in enumerate(self.strong_cols) if strong]
        self.M = len(self.match_col_idx)

        # State list: S, M1..M_M, D1..D_M, I0..I_M, E
        self.states = ['S'] + [f'M{k+1}' for k in range(self.M)] + \
                      [f'D{k+1}' for k in range(self.M)] + \
                      [f'I{k}' for k in range(self.M+1)] + ['E']
        self.state_idx = {s: i for i, s in enumerate(self.states)}

        # Initialize A, B, π
        self._initialize_parameters()

    def _initialize_parameters(self):
        S = len(self.states)
        self.A = np.zeros((S, S))
        # Emission B with Dirichlet prior Π
        self.B = {
            s: {sym: self.pseudocount * self.symbol_pi.get(sym, 0.0)
                for sym in ALPHABET if s.startswith(('M', 'I'))}
            for s in self.states}
        # Start distribution
        self.pi = np.zeros(S)
        self.pi[self.state_idx['S']] = 1.0

        # Transitions out of S (to M1, D1, I0)
        first = self.match_col_idx[0]
        col0 = [seq[first] for seq in self.aligned_sequences]
        gap0 = col0.count('_'); non_gap0 = self.N - gap0
        raw = {'M1': non_gap0/self.N, 'D1': gap0/self.N, 'I0': self.pseudocount}
        total0 = sum(raw.values())
        for tgt, v in raw.items():
            self.A[self.state_idx['S'], self.state_idx[tgt]] = v/total0

        # For each match index, set emissions and transitions
        for k, col_idx in enumerate(self.match_col_idx):
            m = f'M{k+1}'; d = f'D{k+1}'; i = f'I{k}'
            # Emissions for M
            col = [seq[col_idx] for seq in self.aligned_sequences]
            cnt = {sym: col.count(sym) for sym in ALPHABET if sym!='_'}
            for sym in cnt:
                self.B[m][sym] += cnt[sym]
            # Normalize B[m]
            sB = sum(self.B[m].values())
            for sym in self.B[m]: self.B[m][sym] /= sB

            # Next transitions
            if k < self.M-1:
                nxt = self.match_col_idx[k+1]
                coln = [seq[nxt] for seq in self.aligned_sequences]
                gapn = coln.count('_'); non_gapn = self.N-gapn
                raw = {'M'+str(k+2): non_gapn/self.N, 'D'+str(k+2): gapn/self.N, 'I'+str(k+1): self.pseudocount}
                tot = sum(raw.values())
                for tgt, v in raw.items():
                    self.A[self.state_idx[m], self.state_idx[tgt]] = v/tot
                    self.A[self.state_idx[d], self.state_idx[tgt]] = v/tot
            else:
                self.A[self.state_idx[m], self.state_idx['E']] = 1.0
                self.A[self.state_idx[d], self.state_idx['E']] = 1.0

            # Insert self-loop and to next M
            self.A[self.state_idx[i], self.state_idx[i]] = 0.5
            self.A[self.state_idx[i], self.state_idx[m]] = 0.5

        # Final insert to E
        self.A[self.state_idx[f'I{self.M}'], self.state_idx['E']] = 1.0

    def viterbi(self, seq: str) -> Tuple[float, List[str]]:
        T, S = len(seq), len(self.states)
        V = np.full((S, T+1), -np.inf)
        ptr = np.zeros((S, T+1), int)
        V[self.state_idx['S'], 0] = 0.0

        for t in range(1, T+1):
            x = seq[t-1]
            for j in range(S):
                st = self.states[j]
                # emission log-prob
                e = 0.0
                if st.startswith(('M', 'I')):
                    e = math.log(self.B[st].get(x, self.pseudocount))
                # find all predecessors with non-zero transition
                preds = [i for i in range(S) if self.A[i, j] > 0]
                if not preds:
                    # no legal path to j
                    V[j, t] = -np.inf
                    ptr[j, t] = 0
                    continue
                # compute best predecessor
                best_val = -np.inf
                best_i = preds[0]
                for i in preds:
                    val = V[i, t-1] + math.log(self.A[i, j]) + e
                    if val > best_val:
                        best_val, best_i = val, i
                V[j, t] = best_val
                ptr[j, t] = best_i

        # backtrack from end state
        end = self.state_idx['E']
        score = V[end, T]
        path = []
        cur = end
        t = T
        while t > 0:
            path.append(self.states[cur])
            cur = ptr[cur, t]
            t -= 1
        path.append('S')
        return score, list(reversed(path))

    def train(self, sequences: List[str], batch_size: int = 10, verbose: bool = False):
        """
        Viterbi training with fine-tuning: accept batch update only if total log-score increases.
        Uses background Π for emission prior.
        """
        # Precompute baseline score
        base_score = sum(self.viterbi(s)[0] for s in sequences)
        if verbose:
            print(f"Initial total score: {base_score:.3f}")

        # shuffle into batches
        seqs = sequences.copy()
        random.shuffle(seqs)
        batches = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]
        for idx, batch in enumerate(batches, 1):
            if verbose:
                print(f"Batch {idx}/{len(batches)}")
            # save old parameters
            A_old = self.A.copy()
            B_old = {s: self.B[s].copy() for s in self.B}
            # count on batch
            A_cnt = np.zeros_like(self.A)
            B_cnt = {
                s: {sym: self.pseudocount * self.symbol_pi.get(sym, 0)
                    for sym in ALPHABET if s.startswith(('M', 'I'))}
                for s in self.B
            }
            for seq in batch:
                _, path = self.viterbi(seq)
                prev = path[0]
                pos = 0
                for st in path[1:]:
                    A_cnt[self.state_idx[prev], self.state_idx[st]] += 1
                    if st.startswith(('M', 'I')):
                        B_cnt[st][seq[pos]] += 1
                        pos += 1
                    prev = st
            # propose new A, B
            A_new = np.zeros_like(self.A)
            for i in range(len(self.states)):
                row_sum = A_cnt[i].sum()
                if row_sum > 0:
                    A_new[i] = A_cnt[i] / row_sum
                else:
                    A_new[i] = self.A[i]
            B_new = {}
            for st, counts in B_cnt.items():
                total = sum(counts.values())
                B_new[st] = {sym: counts[sym] / total for sym in counts}
            # apply proposal
            self.A, self.B = A_new, {**self.B, **B_new}
            # recompute global score
            new_score = sum(self.viterbi(s)[0] for s in sequences)
            if verbose:
                print(f"  score -> {new_score:.3f}")
            if new_score >= base_score:
                base_score = new_score
                if verbose:
                    print("  accepted")
            else:
                self.A, self.B = A_old, B_old
                if verbose:
                    print("  rejected batch update")

    def pretty_print(self, threshold: float = 0.01):
        print("HMM states:", self.states)
        print("Transitions (p>=%.2f):" % threshold)
        for i, s in enumerate(self.states):
            outs = [f"{s}->{self.states[j]}({self.A[i,j]:.3f})"
                   for j in range(len(self.states)) if self.A[i,j] >= threshold]
            if outs:
                print("  " + ", ".join(outs))


if __name__ == '__main__':
    msa = [str(r.seq) for r in SeqIO.parse('auxiliary\\custom_msa.fasta', 'fasta')]
    dataB = [str(r.seq) for r in SeqIO.parse('auxiliary\\datasetB.fasta', 'fasta')]
    model = HMMProfile(msa)
    model.pretty_print(0.05)
    model.train(dataB, batch_size=10, verbose=True)
    print("Done")
