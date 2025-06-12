from typing import List, Tuple, Dict, Set
import math
from Bio import SeqIO, AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.motifs import Motif
import os
import sys


def align_seq(
    seq1: str,
    seq2: str,
    lookup: Dict[Tuple[str, str], float],
    gap_penalty: float = 2.0
) -> Tuple[float, List[List[float]], str, str]:
    """
    Υλοποίηση του αλγόριθμου καθολικής στοίχισης όπως περιγράφεται στην εργασία.
    Parameters:
        seq1: Πρώτη ακολουθία
        seq2: Δεύτερη ακολουθία
        lookup: Πίνακας αναζήτησης για τα match/mismatch scores
        gap_penalty: Ποινή για την εισαγωγή κενών
    Returns:
        score: Τελικό σκορ της στοίχισης
        dp: Πίνακας DP
        aligned1: Πρώτη στοιχισμένη ακολουθία
        aligned2: Δεύτερη στοιχισμένη ακολουθία

    """
    n, m = len(seq1), len(seq2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = lookup.get((seq1[i - 1], seq2[j - 1]), -1.0)
            dp[i][j] = max(
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j] - gap_penalty,
                dp[i][j - 1] - gap_penalty
            )

    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        score = dp[i][j]
        if i > 0 and j > 0 and score == dp[i - 1][j - 1] + lookup.get((seq1[i - 1], seq2[j - 1]), -1.0):
            aligned1.append(seq1[i - 1]); aligned2.append(seq2[j - 1]); i -= 1; j -= 1
        elif i > 0 and score == dp[i - 1][j] - gap_penalty:
            aligned1.append(seq1[i - 1]); aligned2.append('_'); i -= 1
        else:
            aligned1.append('_'); aligned2.append(seq2[j - 1]); j -= 1

    return dp[n][m], dp, ''.join(reversed(aligned1)), ''.join(reversed(aligned2))


def compute_kmer_freq(seq: str, k: int) -> Dict[str, int]:
    """
    Υπολογίζει τις συχνότητες των k-mers σε μια ακολουθία.
    Parameters:
        seq: Ακολουθία
        k: Μήκος του k-mer
    Returns:
        freqs: Λεξικό με τις συχνότητες των k-mers
    """
    freqs: Dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        freqs[kmer] = freqs.get(kmer, 0) + 1
    return freqs


def cosine_distance(f1: Dict[str, int], f2: Dict[str, int]) -> float:
    """
    Υπολογίζει την απόσταση συνημιτόνου μεταξύ δύο λεξικών συχνοτήτων k-mers.
    Parameters:
        f1: Πρώτο λεξικό συχνοτήτων
        f2: Δεύτερο λεξικό συχνοτήτων
    Returns:
        Απόσταση συνημιτόνου
    """
    keys = set(f1) | set(f2)
    dot = sum(f1.get(k, 0) * f2.get(k, 0) for k in keys)
    norm1 = math.sqrt(sum(v * v for v in f1.values()))
    norm2 = math.sqrt(sum(v * v for v in f2.values()))
    return 1.0 if norm1 == 0 or norm2 == 0 else 1.0 - dot / (norm1 * norm2)


def compute_distance_matrix(seqs: List[str], k: int) -> List[List[float]]:
    """
    Υπολογίζει τον πίνακα αποστάσεων μεταξύ των ακολουθιών με βάση τις συχνότητες k-mers.
    Parameters:
        seqs: Λίστα ακολουθιών
        k: Μήκος του k-mer
    Returns:
        Πίνακας αποστάσεων
    """
    n = len(seqs)
    freqs = [compute_kmer_freq(s, k) for s in seqs]
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(freqs[i], freqs[j])
            mat[i][j] = mat[j][i] = d
    return mat


def propagate_alignment(aligned_cons: str, aligned_seqs: List[str]) -> List[str]:
    """
    Πράττει gap propagation για τις στοιχισμένες ακολουθίες με βάση την ακολουθία συναίνεσης.
    Parameters:
        aligned_cons: Ακολουθία συναίνεσης
        aligned_seqs: Λίστα στοιχισμένων ακολουθιών
    Returns:
        new_aligned: Λίστα με τις νέες στοιχισμένες ακολουθίες
    
    """
    new_aligned: List[str] = []
    for seq in aligned_seqs:
        res = []
        idx = 0
        for c in aligned_cons:
            if c == '_':
                res.append('_')
            else:
                res.append(seq[idx])
                idx += 1
        new_aligned.append(''.join(res))
    return new_aligned


def consensus_sequence(
    aln1: str,
    aln2: str,
    lookup: Dict[Tuple[str, str], float],
    symbol_map: Dict[str, Set[str]],
    next_symbol_id: List[int],
    next_symbol_score: float = 0.5
) -> str:
    """
    Δημιουργεί μια ακολουθία συναίνεσης από δύο στοιχισμένες ακολουθίες.
    Parameters:
        aln1: Πρώτη στοιχισμένη ακολουθία
        aln2: Δεύτερη στοιχισμένη ακολουθία
        lookup: Πίνακας αναζήτησης για τα match/mismatch scores
        symbol_map: Χάρτης συμβόλων
        next_symbol_id: Επόμενο ID συμβόλου
        next_symbol_score: Επόμενο σκορ συμβόλου
    Returns:
        cons: Ακολουθία συναίνεσης
    """
    cons: List[str] = []
    for a, b in zip(aln1, aln2):
        if a == b:
            cons.append(a)
        elif a == '_':
            cons.append(b)
        elif b == '_':
            cons.append(a)
        else:
            for sym, comps in symbol_map.items():
                if a in comps and b in comps:
                    cons.append(sym)
                    break
            else:
                for sym, comps in symbol_map.items():
                    if a in comps or b in comps:
                        comps.update({a, b})
                        for x in comps:
                            lookup[(sym, x)] = lookup[(x, sym)] = next_symbol_score
                        cons.append(sym)
                        break
                else:
                    sym = f"Z{next_symbol_id[0]}"
                    next_symbol_id[0] += 1
                    symbol_map[sym] = {a, b}
                    for x in (a, b):
                        lookup[(sym, x)] = lookup[(x, sym)] = next_symbol_score
                    cons.append(sym)
    return ''.join(cons)


def hierarchical_msa(seqs: List[str], k: int = 3) -> List[str]:
    """
    Υλοποίηση του αλγόριθμου καθολικής στοίχισης με ιεραρχική προσέγγιση.
    Parameters:
        seqs: Λίστα ακολουθιών
        k: Μήκος του k-mer
    Returns:
        results: Λίστα με τις στοιχισμένες ακολουθίες
    """
    lookup: Dict[Tuple[str, str], float] = {}
    symbol_map: Dict[str, Set[str]] = {}
    for x in "ACGT":
        symbol_map[x] = {x}
        for y in "ACGT":
            lookup[(x, y)] = 1.0 if x == y else -1.0
    next_symbol_id = [1]

    clusters = [
        {'members': [i], 'aligned_seqs': [s], 'consensus': s}
        for i, s in enumerate(seqs)
    ]
    results: List[str] = [None] * len(seqs)

    while len(clusters) > 1:
        cons_list = [c['consensus'] for c in clusters]
        dist = compute_distance_matrix(cons_list, k)
        i_min, j_min = min(
            ((i, j) for i in range(len(dist)) for j in range(i + 1, len(dist))),
            key=lambda ij: dist[ij[0]][ij[1]]
        )
        cA, cB = clusters[i_min], clusters[j_min]
        _, _, alnA, alnB = align_seq(cA['consensus'], cB['consensus'], lookup)
        alignedA = propagate_alignment(alnA, cA['aligned_seqs'])
        alignedB = propagate_alignment(alnB, cB['aligned_seqs'])

        for idx, seq in zip(cB['members'], alignedB):
            results[idx] = seq

        new_cons = consensus_sequence(alnA, alnB, lookup, symbol_map, next_symbol_id)
        merged = {
            'members': cA['members'] + cB['members'],
            'aligned_seqs': alignedA + alignedB,
            'consensus': new_cons
        }
        clusters[i_min] = merged
        del clusters[j_min]

    final = clusters[0]
    for idx, seq in zip(final['members'], final['aligned_seqs']):
        results[idx] = seq
    return results


def percent_identity(seq1: str, seq2: str) -> float:
    """
    Υπολογίζει το ποσοστό ταυτότητας μεταξύ δύο ακολουθιών.
    Parameters:
        seq1: Πρώτη ακολουθία
        seq2: Δεύτερη ακολουθία
    Returns:
        Ποσοστό ταυτότητας
    """
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1) * 100.0


def compute_sp_cs(
    aln_test: List[str],
    aln_ref: List[str],
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_penalty: float = -2.0
) -> Tuple[float, float]:
    """
    Υπολογίζει το σκορ SP και CS μεταξύ δύο στοιχισμένων ακολουθιών.
    Parameters:
        aln_test: Στοιχισμένη ακολουθία προς αξιολόγηση
        aln_ref: Στοιχισμένη ακολουθία αναφοράς
        match_score: Σκορ για ταυτοχρονισμένα στοιχεία
        mismatch_score: Σκορ για μη ταυτοχρονισμένα στοιχεία
        gap_penalty: Ποινή για κενά
    Returns:
        sp: Σκορ SP
        cs: Σκορ CS
    """
    if len(aln_test) != len(aln_ref):
        raise ValueError("Both alignments must have the same number of sequences")
    k = len(aln_test)
    L = len(aln_ref[0])
    sp = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            for c in range(L):
                a, b = aln_test[i][c], aln_test[j][c]
                if a in ('-', '_') or b in ('-', '_'):
                    sp += gap_penalty
                elif a == b:
                    sp += match_score
                else:
                    sp += mismatch_score
    identical = 0
    for c in range(L):
        if tuple(seq[c] for seq in aln_test) == tuple(seq[c] for seq in aln_ref):
            identical += 1
    cs = identical / L
    return sp, cs

if __name__ == "__main__":
    fasta_path = "datasetA.fasta"
    seqs = [str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")]

    msa_custom_list = hierarchical_msa(seqs, k=3)

    print("=== Custom MSA ===")
    for i, s in enumerate(msa_custom_list):
        print(f">seq{i+1}\n{s}")
    with open( "custom_msa.fasta" , "w" ) as f:
        for i, s in enumerate(msa_custom_list):
            f.write(f">seq{i+1}\n{s}\n")

    clustal_path = "clustalw.aln"
    if not os.path.exists ( clustal_path ) :
        print ( f"\nNo '{clustal_path}' file found; skipping benchmarking." )
        sys.exit ( 0 )
    msa_ref_old = AlignIO.read(clustal_path, "clustal")
    msa_ref = msa_ref_old.alignment

    motif_ref = Motif(alignment=msa_ref, alphabet="ACGT")
    motif_test = Motif(
        alignment=MultipleSeqAlignment(
            [SeqRecord(Seq(s), id=f"C{i+1}") for i, s in enumerate(msa_custom_list)]
        ).alignment,
        alphabet="ACGT"
    )
    cons_ref, cons_test = str(motif_ref.consensus), str(motif_test.consensus)

    lookup_simple = {(x, y): 1.0 if x == y else 0.0 for x in "ACGT" for y in "ACGT"}
    _, _, aln_cons_ref, aln_cons_test = align_seq(cons_ref, cons_test, lookup_simple)

    pid = percent_identity(aln_cons_ref, aln_cons_test)

    print("=== Consensus Alignment & PID ===")
    print(aln_cons_ref)
    print(aln_cons_test)
    print(f"PID: {pid:.2f}%")

    ref_raw = [str(rec.seq).replace('-', '_') for rec in msa_ref_old]
    test_raw = msa_custom_list
    ref_prop = propagate_alignment(aln_cons_ref, ref_raw)
    test_prop = propagate_alignment(aln_cons_test, test_raw)

    sp_score , cs_fraction = compute_sp_cs ( test_prop , ref_prop )

    print ( "\n=== Raw SP and CS ===" )
    print ( f"Sum-of-Pairs: {sp_score:.1f}" )
    print ( f"Column Score: {cs_fraction:.3f}" )

    k = len ( test_prop )
    L = len ( test_prop [ 0 ] )
    n_pairs = k * (k - 1) // 2

    max_sp = n_pairs * L * 1.0

    sp_norm = sp_score / max_sp
    sp_pct = sp_norm * 100.0

    cs_pct = cs_fraction * 100.0

    print ( "\n=== Normalized Metrics ===" )
    print ( f"Normalized SP   : {sp_norm:.3f} ({sp_pct:.1f}%)" )
    print ( f"Normalized CS   : {cs_fraction:.3f} ({cs_pct:.1f}%)" )