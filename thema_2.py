from typing import List, Tuple, Dict, Set
import math


def align_seq(
    seq1: str,
    seq2: str,
    lookup: Dict[Tuple[str, str], float],
    gap_penalty: float = -2.0
) -> Tuple[float, List[List[float]], str, str]:
    """
    Perform custom global alignment between seq1 and seq2 using a lookup table for scores.

    :param seq1: First sequence
    :param seq2: Second sequence
    :param lookup: Scoring lookup table for pairs
    :param gap_penalty: Penalty for gaps
    :return: (score, DP matrix, aligned seq1, aligned seq2)
    """
    n, m = len(seq1), len(seq2)
    # Initialize DP matrix
    dp: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1): dp[0][j] = dp[0][j-1] + gap_penalty

    # Fill DP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i-1][j-1] + lookup.get((seq1[i-1], seq2[j-1]), -1.0)
            up   = dp[i-1][j]   + gap_penalty
            left = dp[i][j-1]   + gap_penalty
            dp[i][j] = max(diag, up, left)

    # Backtracking
    aligned1, aligned2 = [], []
    i, j = n, m
    while i>0 or j>0:
        current = dp[i][j]
        if i>0 and j>0 and current == dp[i-1][j-1] + lookup.get((seq1[i-1], seq2[j-1]), -1.0):
            aligned1.append(seq1[i-1]); aligned2.append(seq2[j-1]); i-=1; j-=1
        elif i>0 and current == dp[i-1][j] + gap_penalty:
            aligned1.append(seq1[i-1]); aligned2.append('_'); i-=1
        else:
            aligned1.append('_'); aligned2.append(seq2[j-1]); j-=1

    return dp[n][m], dp, ''.join(reversed(aligned1)), ''.join(reversed(aligned2))


def compute_kmer_freq(seq: str, k: int) -> Dict[str, int]:
    """
    Compute k-mer frequencies for a given sequence.
    :param seq:  Input sequence
    :param k:  Length of k-mers
    :return:  Dictionary of k-mer frequencies
    """
    freqs: Dict[str,int] = {}
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        freqs[kmer] = freqs.get(kmer, 0) + 1
    return freqs


def cosine_distance(freq1: Dict[str,int], freq2: Dict[str,int]) -> float:
    """
    Compute cosine distance between two frequency dictionaries.
    :param freq1:  frequency dictionary for first sequence
    :param freq2:  frequency dictionary for second sequence
    :return:  cosine distance
    """
    keys = set(freq1)|set(freq2)
    dot = sum(freq1.get(k,0)*freq2.get(k,0) for k in keys)
    norm1 = math.sqrt(sum(v*v for v in freq1.values()))
    norm2 = math.sqrt(sum(v*v for v in freq2.values()))
    return 1.0 if norm1==0 or norm2==0 else 1.0 - dot/(norm1*norm2)


def compute_distance_matrix(seqs: List[str], k: int) -> List[List[float]]:
    """
    Compute distance matrix for a list of sequences using k-mer frequencies.
    :param seqs:  List of sequences
    :param k:  Length of k-mers
    :return:  Distance matrix
    """
    n = len(seqs)
    mat = [[0.0]*n for _ in range(n)]
    freqs = [compute_kmer_freq(s,k) for s in seqs]
    for i in range(n):
        for j in range(i+1,n):
            d = cosine_distance(freqs[i], freqs[j]); mat[i][j]=mat[j][i]=d
    return mat


def propagate_alignment(
    aligned_cons: str,
    aligned_seqs: List[str]
) -> List[str]:
    """
    Propagate gaps from aligned_cons onto each sequence in aligned_seqs.

    :param aligned_cons: Aligned consensus sequence (with '_' where gaps were inserted)
    :param aligned_seqs: List of sequences that were previously aligned to the old consensus
    :return: List of sequences re-aligned to the new consensus
    """
    new_aligned = []
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
    lookup: Dict[Tuple[str,str],float],
    symbol_map: Dict[str,Set[str]],
    next_symbol_id: List[int],
    next_symbol_score: float = 0.5
) -> str:
    """
    Create consensus of two aligned sequences, updating lookup and symbol_map.
    :param aln1: Aligned sequence 1
    :param aln2: Aligned sequence 2
    :param lookup: Lookup table for scoring
    :param symbol_map: Map of symbols to their components
    :param next_symbol_id: Next available symbol ID
    :param next_symbol_score: Score for new symbols
    :return: Consensus sequence
    """
    cons=[]
    for a,b in zip(aln1,aln2):
        if a==b: cons.append(a)
        elif a=='_': cons.append(b)
        elif b=='_': cons.append(a)
        else:
            # existing composite?
            for sym,comps in symbol_map.items():
                if a in comps and b in comps:
                    cons.append(sym); break
            else:
                # extend or new
                for sym,comps in symbol_map.items():
                    if a in comps or b in comps:
                        comps.update({a,b})
                        for x in comps: lookup[(sym,x)]=lookup[(x,sym)]=next_symbol_score
                        cons.append(sym); break
                else:
                    sym=f"Z{next_symbol_id[0]}"; next_symbol_id[0]+=1
                    symbol_map[sym]={a,b}
                    for x in (a,b): lookup[(sym,x)]=lookup[(x,sym)]=next_symbol_score
                    cons.append(sym)
    return ''.join(cons)


def hierarchical_msa(seqs: List[str], k: int = 3) -> List[str]:
    """
    Hierarchical MSA producing aligned original sequences.
    :param seqs: List of sequences to align
    :param k: Length of k-mers for distance computation
    :return: List of aligned sequences
    """
    # initialize lookup and symbol_map
    lookup: Dict [ Tuple [ str , str ] , float ] = {}
    symbol_map: Dict [ str , Set [ str ] ] = {}
    for x in "ACGT" :
        symbol_map [ x ] = {x}
        for y in "ACGT" :
            lookup [ (x , y) ] = 1.0 if x == y else 0.0
    next_symbol_id = [ 1 ]

    # initialize clusters: each has members, aligned_seqs, consensus
    clusters = [ ]
    for idx , seq in enumerate ( seqs ) :
        clusters.append ( {
            'members' : [ idx ] ,
            'aligned_seqs' : [ seq ] ,
            'consensus' : seq
        } )
    # prepare results placeholder
    results: List [ str ] = [ None ] * len ( seqs )

    # merge until one cluster remains
    while len ( clusters ) > 1 :
        # compute distance matrix on consensus strings
        cons_list = [ c [ 'consensus' ] for c in clusters ]
        dist = compute_distance_matrix ( cons_list , k )
        # find closest pair
        i_min , j_min = 0 , 1
        min_d = dist [ 0 ] [ 1 ]
        for i in range ( len ( clusters ) ) :
            for j in range ( i + 1 , len ( clusters ) ) :
                if dist [ i ] [ j ] < min_d :
                    i_min , j_min , min_d = i , j , dist [ i ] [ j ]
        cA , cB = clusters [ i_min ] , clusters [ j_min ]
        # align their consensus
        _ , _ , alnA , alnB = align_seq ( cA [ 'consensus' ] , cB [ 'consensus' ] , lookup )
        # propagate alignment to all sequences in cA and cB
        alignedA = propagate_alignment ( alnA , cA [ 'aligned_seqs' ] )
        alignedB = propagate_alignment ( alnB , cB [ 'aligned_seqs' ] )
        # record aligned sequences of cluster B (to be removed)
        for idx , seq in zip ( cB [ 'members' ] , alignedB ) :
            results [ idx ] = seq
        # compute new consensus for merged cluster
        new_cons = consensus_sequence ( alnA , alnB , lookup , symbol_map , next_symbol_id )
        # create merged cluster with updated members, aligned_seqs, consensus
        merged_members = cA [ 'members' ] + cB [ 'members' ]
        merged_aligned = alignedA + alignedB
        clusters [ i_min ] = {
            'members' : merged_members ,
            'aligned_seqs' : merged_aligned ,
            'consensus' : new_cons
        }
        # remove the B cluster
        del clusters [ j_min ]
    # final cluster: record aligned sequences
    final_cluster = clusters [ 0 ]
    for idx , seq in zip ( final_cluster [ 'members' ] , final_cluster [ 'aligned_seqs' ] ) :
        results [ idx ] = seq

    # all results should now be filled and aligned to the same length
    return results


if __name__=="__main__":
    datasetA=[]
    with open("auxiliary/datasetA.txt") as f:
        for line in f:
            if line.startswith("seq"):
                datasetA.append(line.split(":")[1].strip())

    msa=hierarchical_msa(datasetA,k=3)
    for seq in msa: print(seq)