from typing import List, Tuple
import random
import math
import numpy as np

# αλφάβητο
ALPHABET = ['A', 'C', 'G', 'T', '_']

class HMMProfile:
    """
    Κλάση για την εκπαίδευση και την εφαρμογή ενός μοντέλου HMM για στοίχιση ακολουθιών.
    """
    @staticmethod
    def entropy(column: List[str]) -> float:
        """
        Υπολογίζει την εντροπία μιας στήλης ακολουθιών.

        Args:
            column (List[str]): Στήλη ακολουθιών.

        Returns:
            float: Η εντροπία της στήλης.
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

    @staticmethod
    def information_content(column: List[str]) -> float:
        """
        Υπολογίζει το περιεχόμενο πληροφορίας μιας στήλης ακολουθιών.
        Args:
            column (List[str]): Στήλη ακολουθιών.
        Returns:
            float: Το περιεχόμενο πληροφορίας της στήλης.
        """
        return math.log2(len(ALPHABET)) - HMMProfile.entropy(column)

    def __init__(self,
                 aligned_sequences: List[str],
                 ic_threshold: float = 1.0,
                 pseudocount: float = 1e-6):
        """
        Αρχικοποιεί το HMMProfile με τις δοσμένες ευθυγραμμισμένες ακολουθίες.
        Args:
            aligned_sequences (List[str]): Λίστα ευθυγραμμισμένων ακολουθιών.
            ic_threshold (float): Κατώφλι πληροφορίας για την κατηγοριοποίηση στηλών.
            pseudocount (float): Ψευδομετρητής για τις εκτιμήσεις πιθανοτήτων.
        """
        self.aligned_sequences = aligned_sequences
        self.N = len(aligned_sequences)
        self.L = len(aligned_sequences[0])
        self.ic_threshold = ic_threshold
        self.pseudocount = pseudocount

        # Υπολογισμός της κατανομής των συμβόλων στο alignment με το οποίο αρχικοποιείται το μοντέλο
        counts = {sym: 0 for sym in ALPHABET}
        for seq in aligned_sequences:
            for ch in seq:
                counts[ch] += 1
        total = sum(counts[s] for s in ALPHABET if s != '_')
        self.theta = {sym: counts.get(sym, 0) / total for sym in ALPHABET}

        # Υπολογισμός ισχυρών και αδύναμων καταστάσεων
        cols = list(zip(*aligned_sequences))
        self.strong_cols = [self.information_content(list(col)) > ic_threshold for col in cols]
        self.match_col_idx = [i for i, strong in enumerate(self.strong_cols) if strong]
        self.weak_col_idx = [i for i, strong in enumerate(self.strong_cols) if not strong]
        self.M = len(self.match_col_idx)

        # States
        self.states = (['S']
                       + [f'M{k+1}' for k in range(self.M)]
                       + [f'D{k+1}' for k in range(self.M)]
                       + [f'I{k}'   for k in range(self.M+1)]
                       + ['E'])
        self.state_idx = {s: i for i, s in enumerate(self.states)}

        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Αρχικοποιεί τους πίνακες πιθανοτήτων A και Β.
        """
        S = len(self.states)
        self.A = np.zeros((S, S))

        # Εκπομπές (emissions) για τα M και I states
        self.B = {s: {sym: self.pseudocount * self.theta[sym] for sym in ALPHABET}
                    for s in self.states if s.startswith(('M', 'I'))}
        for s in self.states:
            self.B.setdefault(s, {})
        n_ins = self.M + 1
        # --- Αρχικοποίηση καταστάσεων εισαγωγής (insert states) για τις αδύναμες καταστάσεις---
        for k, col_idx in enumerate(self.weak_col_idx[:n_ins]):
            ik = f'I{k}'
            # μέτρηση των συχνοτήτων για την αδύναμη στήλη
            colw = [seq[col_idx] for seq in self.aligned_sequences]
            cntw = {sym: colw.count(sym) for sym in ALPHABET if sym != '_'}
            # προσθήκη των συχνοτήτων στο Pseudocount
            for sym, c in cntw.items():
                self.B[ik][sym] += c
            # normalize
            totB = sum(self.B[ik].values())
            for sym in ALPHABET:
                self.B[ik][sym] /= totB

        # Helper για τη κανονικοποίηση των πιθανοτήτων
        def normalize(raw):
            tot = sum(raw.values())
            return {k: v / tot for k, v in raw.items()}

        # Αρχικοποίηση μεταβάσεων από την κατάσταση S και I0
        first = self.match_col_idx[0]
        col0 = [seq[first] for seq in self.aligned_sequences]
        gap0 = col0.count('_'); non0 = self.N - gap0
        raw0 = {'M1': non0/self.N, 'D1': gap0/self.N, 'I0': self.pseudocount}
        trans0 = normalize(raw0)
        for tgt, p in trans0.items():
            self.A[self.state_idx['S'], self.state_idx[tgt]] = p
            self.A[self.state_idx['I0'], self.state_idx[tgt]] = p

        # Για κάθε κατάσταση M, υπολογίζουμε τις εκπομπές και τις μεταβάσεις
        for k, col_idx in enumerate(self.match_col_idx):
            m = f'M{k+1}'; d = f'D{k+1}'; i = f'I{k}'
            # εκπομπές για τη κ-κοστή στήλη match
            col = [seq[col_idx] for seq in self.aligned_sequences]
            cnt = {sym: col.count(sym) for sym in ALPHABET if sym != '_'}
            for sym, c in cnt.items(): self.B[m][sym] += c
            totM = sum(self.B[m].values())
            for sym in ALPHABET: self.B[m][sym] /= totM

            # μεταβάσεις από το κ-ακοστό M state στο αντίστοιχο D και I state
            if k < self.M-1:
                nxt = self.match_col_idx[k+1]
                coln = [seq[nxt] for seq in self.aligned_sequences]
                gapn = coln.count('_'); nonn = self.N - gapn
                raw = {f'M{k+2}': nonn/self.N, f'D{k+2}': gapn/self.N, f'I{k+1}': self.pseudocount}
                for src in (m, d):
                    for tgt, p in normalize(raw).items():
                        self.A[self.state_idx[src], self.state_idx[tgt]] = p
            else:
                for src in (m, d): self.A[self.state_idx[src], self.state_idx['E']] = 1.0

            # I_k transitions based on weak column if exists, else pseudocounts
            # Μεταβάσεις στη κ-ακοστή Ι state αν υπάρχει η αδύναμη στήλη, αλλιώς η μετάβαση ορίζεται με το pseudocount
            idx_i = self.state_idx[i]
            if k < len(self.weak_col_idx):
                wcol = self.weak_col_idx[k]
                colw = [seq[wcol] for seq in self.aligned_sequences]
                gapw = colw.count('_'); nonw = self.N - gapw
                rawI = {i: self.pseudocount, f'M{k+1}': nonw/self.N, f'D{k+1}': gapw/self.N}
            else:
                rawI = {i: self.pseudocount, f'M{k+1}': self.pseudocount, f'D{k+1}': self.pseudocount}
            for tgt, p in normalize(rawI).items():
                self.A[idx_i, self.state_idx[tgt]] = p

        # Final insert to end
        last_i = f'I{self.M}'
        self.A[self.state_idx[last_i], self.state_idx['E']] = 1.0

    # viterbi, train, pretty_print, format_alignment unchanged


    def viterbi(self, seq: str) -> Tuple[float, List[str]]:
        """
        Υλοποιεί τον αλγόριθμο Viterbi για την εύρεση της πιο πιθανής διαδρομής

        Args:
            seq (str): Ακολουθία εισόδου.

        Returns:
            Tuple[float, List[str]]: Η πιθανότητα της καλύτερης διαδρομής και η ίδια η διαδρομή (best path).
        """
        T, S = len(seq), len(self.states)
        V = np.full((S, T+1), -np.inf); ptr = np.zeros((S, T+1), int)
        V[self.state_idx['S'], 0] = 0.0
        for t in range(1, T+1):
            x = seq[t-1]
            for j in range(S):
                st = self.states[j]; e = math.log(self.B[st].get(x, self.pseudocount)) if st.startswith(('M','I')) else 0.0
                preds = [i for i in range(S) if self.A[i,j]>0]
                if not preds: continue
                best = max(preds, key=lambda i: V[i,t-1]+math.log(self.A[i,j])+e)
                V[j,t] = V[best,t-1]+math.log(self.A[best,j])+e; ptr[j,t]=best
        score = V[self.state_idx['E'],T]; path=[]; cur=self.state_idx['E']; t=T
        while t>0: path.append(self.states[cur]); cur=ptr[cur,t]; t-=1
        path.append('S'); return score, list(reversed(path))


    def train(self, sequences: List[str], batch_size: int = 10, verbose: bool = False):
        """
        Εκπαιδεύει το HMMProfile με τις δοσμένες ακολουθίες χρησιμοποιώντας τον αλγόριθμο EM.
        Η εκπαίδευση γίνεται σε παρτίδες για να επιταχυνθεί η διαδικασία.

        Args:
            sequences (List[str]): Λίστα ακολουθιών.
            batch_size (int, optional): Μέγεθος παρτίδας. Defaults to 10.
            verbose (bool, optional): Ενεργοποίηση εκτύπωσης λεπτομερειών. Defaults to False.
        """
        base_score = sum(self.viterbi(s)[0] for s in sequences)
        if verbose:
            print(f"Initial total score: {base_score:.3f}")

        seqs = sequences.copy()
        random.shuffle(seqs)
        batches = [seqs[i:i+batch_size] for i in range(0, len(seqs), batch_size)]

        for idx, batch in enumerate(batches, 1):
            if verbose:
                print(f"Batch {idx}/{len(batches)}")
            A_old = self.A.copy()
            B_old = {s: self.B[s].copy() for s in self.B}

            A_cnt = np.zeros_like(self.A)
            B_cnt = {
                s: {sym: self.pseudocount * self.theta[sym]
                    for sym in ALPHABET}
                for s in self.B
            }

            for seq in batch:
                _, path = self.viterbi(seq)
                prev, pos = path[0], 0
                for st in path[1:]:
                    A_cnt[self.state_idx[prev], self.state_idx[st]] += 1
                    if st.startswith(('M', 'I')):
                        B_cnt[st][seq[pos]] += 1
                        pos += 1
                    prev = st

            # A'
            A_new = np.zeros_like(self.A)
            for i in range(len(self.states)):
                row_sum = A_cnt[i].sum()
                A_new[i] = (A_cnt[i] / row_sum) if row_sum > 0 else self.A[i]

            # B'
            B_new = {}
            for st, counts in B_cnt.items():
                tot = sum(counts.values())
                B_new[st] = {sym: counts[sym]/tot for sym in counts}

            # accept/reject
            self.A, self.B = A_new, {**self.B, **B_new}
            new_score = sum(self.viterbi(s)[0] for s in sequences)
            if verbose:
                print(f"  score -> {new_score:.3f}")
            if new_score < base_score:
                self.A, self.B = A_old, B_old
                if verbose:
                    print("  rejected batch update")
            else:
                base_score = new_score
                if verbose:
                    print("  accepted")


    def pretty_print(self, threshold: float = 0.01):
        """
        Εκτυπώνει τις καταστάσεις του HMM και τις μεταβάσεις τους.

        Args:
            threshold (float, optional): Κατώφλι για την εκτύπωση μεταβάσεων. Defaults to 0.01.
        """
        print("HMM states:", self.states)
        print(f"Transitions (p>={threshold:.2f}):")
        for i, s in enumerate(self.states):
            outs = [
                f"{s}->{self.states[j]}({self.A[i,j]:.3f})"
                for j in range(len(self.states))
                if self.A[i,j] >= threshold
            ]
            if outs:
                print("  " + ", ".join(outs))


    def format_alignment(self, seq: str, path: List[str]) -> Tuple[str, str]:
        """
        Μορφοποιεί την ευθυγράμμιση με βάση τη διαδρομή Viterbi.
        Δημιουργεί δύο ακολουθίες: μία με τα ευθυγραμμισμένα σύμβολα και μία με τις σημάνσεις.

        Args:
            seq (str): Ακολουθία εισόδου.
            path (List[str]): Διαδρομή Viterbi.

        Returns:
            Tuple[str, str]: Οι ευθυγραμμισμένες ακολουθίες και οι σημάνσεις.
        """
        aligned, anno, pos = [], [], 0
        for st in path[1:]:
            if st == 'E':
                break
            if st.startswith('M'):
                aligned.append(seq[pos]); anno.append('|'); pos += 1
            elif st.startswith('I'):
                aligned.append(seq[pos]); anno.append('.'); pos += 1
            elif st.startswith('D'):
                aligned.append('_'); anno.append(' ')
        return ''.join(aligned), ''.join(anno)
