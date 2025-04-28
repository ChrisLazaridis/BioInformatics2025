import pickle
from Bio import SeqIO
from hmm import HMMProfile


if __name__ == '__main__':
    msa = [str(r.seq) for r in SeqIO.parse('custom_msa.fasta', 'fasta')]
    dataB = [str(r.seq) for r in SeqIO.parse('datasetB.fasta', 'fasta')]
    model = HMMProfile(msa)
    model.pretty_print(0.05)
    model.train(dataB, batch_size=10, verbose=True)
    with open( '../auxiliary/trained_hmm_profile.pkl' , 'wb' ) as fout:
        pickle.dump(model, fout)
    print("Model serialized to 'trained_hmm_profile.pkl'")
    print("Done")
