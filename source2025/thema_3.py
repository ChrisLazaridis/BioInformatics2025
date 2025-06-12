import pickle
from Bio import SeqIO
from hmm import HMMProfile

# --- helper εκτυπώνει τα 2d arrays με τις αντίστοιχες ετικέτες---
def print_matrix(mat, row_labels, col_labels, fmt="{:.6e}", sep="\t"):
    # header
    header = [""] + [f"{j}:{lbl}" for j, lbl in enumerate(col_labels)]
    print(sep.join(header))
    # each row
    for i, row_lbl in enumerate(row_labels):
        values = [fmt.format(mat[i, j]) for j in range(mat.shape[1])]
        print(sep.join([f"{i}:{row_lbl}"] + values))
    print()

# --- helper εκτυπώνει το πίνακα Β (στη πραγματικότητα είναι dict-of-dict ---
def print_emissions(B, states, symbols, fmt="{:.6e}", sep="\t"):
    # header
    print(sep.join(["state"] + symbols))
    for s in states:
        row = [fmt.format(B.get(s, {}).get(sym, 0.0)) for sym in symbols]
        print(sep.join([s] + row))
    print()

if __name__ == '__main__':
    msa = [str(r.seq) for r in SeqIO.parse('custom_msa.fasta', 'fasta')]
    dataB = [str(r.seq) for r in SeqIO.parse('datasetB.fasta', 'fasta')]
    model = HMMProfile(msa)

    print("State index mapping:")
    for idx, st in enumerate(model.states):
        print(f"  {idx:2d} → {st}")
    print()

    print("Transition matrix A:")
    print_matrix(model.A, model.states, model.states, fmt="{:.6e}")

    print("Emission matrix B:")
    alphabet = ['A','C','G','T','_']
    print_emissions(model.B, model.states, alphabet, fmt="{:.6e}")

    print ( "Background symbol distribution (Θ):" )
    for sym , p in model.theta.items () :
        print ( f"  {sym}\t→ {p:.6e}" )
    print ()

    model.train(dataB, batch_size=10, verbose=True)
    with open('trained_hmm_profile.pkl', 'wb') as fout:
        pickle.dump(model, fout)
    print("Model serialized to 'trained_hmm_profile.pkl'")
    print("Done")
