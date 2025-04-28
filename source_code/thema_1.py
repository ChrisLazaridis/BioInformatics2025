import random
from typing import List

# For reproducability
random.seed(42)

patterns = [ "ATTAGA" , "ACGCATTT" , "AGGACTCAA" , "ATTTCAGT" ]

def mutate_pattern(pattern: str, max_mutations: int = 2) -> str:
    """
    Can apply a maximum of max_mutations to the pattern.
    - Each mutation: 50% substitution, 50% deletion (both in a random position)
    :param pattern: Input pattern
    :param max_mutations: Maximum number of mutations
    :return: Mutated pattern
    """
    pattern_list = list(pattern)
    m = random.randint(0, max_mutations)
    # Επιλογή θέσεων για μετάλλαξη
    positions = random.sample(range(len(pattern_list)), m)
    for pos in positions:
        if random.random() < 0.5:
            # Substitution
            pattern_list[pos] = random.choice(['A', 'C', 'G', 'T'])
        else:
            # Deletion
            pattern_list[pos] = ''
    return ''.join(pattern_list)


def create_random_sequence(min_len: int = 1, max_len: int = 2) -> str:
    """
    Creates a random sequence of A, C, G, T with a random length between min_len and max_len.
    :param min_len: minimum length of the random string
    :param max_len: maximum length of the random string
    :return: Random string of A, C, G, T
    """
    k = random.randint(min_len, max_len)
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=k))

def save_fasta(seqs: List[str], filename: str) -> bool:
    """
    Saves a list of sequences to a FASTA file.
    :param seqs: List of sequences
    :param filename: Name of the output file
    :return: True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            for i, s in enumerate(seqs, 1):
                f.write(f">seq{i}\n{s}\n")
        return True
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
        return False


def generate_sequence ( ) -> str :
    """
    creates a random sequence with a pattern and mutations
    :return: Created sequence
    """
    seq = create_random_sequence ( 1 , 3 )
    for pat in patterns :
        mut = mutate_pattern ( pat , max_mutations = 2 )
        seq += mut
    seq += create_random_sequence ( 1 , 2 )
    return seq
# print statistics for each dataset
def print_statistics(dataset: List[str], name: str):
    """
    Print statistics for a given dataset.
    :param dataset:
    :param name:
    :return:
    """
    print(f"Statistics for {name}:")
    print(f"  Number of sequences: {len(dataset)}")
    print(f"  Length of first sequence: {len(dataset[0])}")
    print(f"  Length of last sequence: {len(dataset[-1])}")
    print(f"  average length: {sum(len(seq) for seq in dataset) / len(dataset):.2f}")
    print(f"  First sequence: {dataset[0]}")
    print(f"  Last sequence: {dataset[-1]}")
    print()
if __name__ == "__main__":
    sequences: List[str] = [generate_sequence() for _ in range(100)]

    # Shuffle και Διαχωρισμός
    random.shuffle(sequences)
    datasetA = sequences[:10]
    datasetB = sequences[10:80]
    datasetC = sequences[80:]

    print(f"Συνολικές αλληλουχίες: {len(sequences)}")
    print(f"datasetA: {len(datasetA)} (π.χ. {datasetA[0]})")
    print(f"datasetB: {len(datasetB)}")
    print(f"datasetC: {len(datasetC)}")

    if save_fasta( datasetA, "datasetA.fasta" ):
        print("datasetA αποθηκεύτηκε σε datasetA.fasta")
    if save_fasta( datasetB, "datasetB.fasta" ):
        print("datasetB αποθηκεύτηκε σε datasetB.fasta")
    if save_fasta( datasetC, "datasetC.fasta" ):
        print("datasetC αποθηκεύτηκε σε datasetC.fasta")


print_statistics(datasetA, "datasetA")
print_statistics(datasetB, "datasetB")
print_statistics(datasetC, "datasetC")

