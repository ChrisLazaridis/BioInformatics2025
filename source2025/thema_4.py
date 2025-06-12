from hmm import HMMProfile
import pickle
from Bio import SeqIO
import random


if __name__ == "__main__":
    with open("trained_hmm_profile.pkl", "rb") as fin:
        model: HMMProfile = pickle.load(fin)


    with open( "alignment.txt" , "w" ) as fout:
        for record in SeqIO.parse("datasetC.fasta", "fasta"):
            seq_str = str(record.seq)
            score, path = model.viterbi(seq_str)

            aligned_seq, annotation = model.format_alignment(seq_str, path)

            print(f">{record.id}")
            print(aligned_seq)
            print(annotation)
            print(f"Score: {score:.3f}")
            print(f"Path : {' '.join(path)}\n")
            fout.write ( f">{record.id}\n" )
            fout.write ( aligned_seq + "\n" )
            fout.write ( annotation + "\n" )
            fout.write ( f"Score: {score:.3f}\n" )
            fout.write ( f"Path : {' '.join ( path )}\n\n" )

    rand_seq = []
    for i in range(20):
        k = random.randint(30, 40)
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=k))
        rand_seq.append(seq)
    with open( "random_alignment.txt" , "w" ) as fout:
        for seq_str in rand_seq:
            score, path = model.viterbi(seq_str)

            aligned_seq, annotation = model.format_alignment(seq_str, path)

            print (f">random_seq\n{seq_str}\n")
            print(f">aligned_seq")
            print(aligned_seq)
            print(annotation)
            print(f"Score: {score:.3f}")
            print(f"Path : {' '.join(path)}\n")
            fout.write( f">random_seq\n{seq_str}\n")
            fout.write ( f">aligned_seq\n" )
            fout.write ( aligned_seq + "\n" )
            fout.write ( annotation + "\n" )
            fout.write ( f"Score: {score:.3f}\n" )
            fout.write ( f"Path : {' '.join ( path )}\n\n" )
