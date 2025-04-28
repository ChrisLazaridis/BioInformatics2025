from hmm import HMMProfile
import pickle
from Bio import SeqIO
import random

random.seed(42)

if __name__ == "__main__":
    # 1. Load trained model
    with open("trained_hmm_profile.pkl", "rb") as fin:
        model: HMMProfile = pickle.load(fin)

    # 2. Iterate over all sequences in datasetC.fasta

    with open( "../auxiliary/alignment.txt" , "w" ) as fout:
        for record in SeqIO.parse("datasetC.fasta", "fasta"):
            seq_str = str(record.seq)
            score, path = model.viterbi(seq_str)

            # 3. Format the alignment
            aligned_seq, annotation = model.format_alignment(seq_str, path)

            # 4. Print nicely
            print(f">{record.id}")
            print(aligned_seq)
            print(annotation)
            print(f"Score: {score:.3f}")
            print(f"Path : {' '.join(path)}\n")
            # 5. Write to file
            fout.write ( f">{record.id}\n" )
            fout.write ( aligned_seq + "\n" )
            fout.write ( annotation + "\n" )
            fout.write ( f"Score: {score:.3f}\n" )
            fout.write ( f"Path : {' '.join ( path )}\n\n" )

    # 6. create 20 fully random sequences
    rand_seq = []
    for i in range(20):
        k = random.randint(30, 40)
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=k))
        rand_seq.append(seq)
    # 7. Iterate over all sequences in rand_seq
    with open( "../auxiliary/random_alignment.txt" , "w" ) as fout:
        for seq_str in rand_seq:
            score, path = model.viterbi(seq_str)

            # 8. Format the alignment
            aligned_seq, annotation = model.format_alignment(seq_str, path)

            # 9. Print nicely
            print (f">random_seq\n{seq_str}\n")
            print(f">aligned_seq")
            print(aligned_seq)
            print(annotation)
            print(f"Score: {score:.3f}")
            print(f"Path : {' '.join(path)}\n")
            # 10. Write to file
            fout.write( f">random_seq\n{seq_str}\n")
            fout.write ( f">aligned_seq\n" )
            fout.write ( aligned_seq + "\n" )
            fout.write ( annotation + "\n" )
            fout.write ( f"Score: {score:.3f}\n" )
            fout.write ( f"Path : {' '.join ( path )}\n\n" )
