import random
from typing import List

# Για αναπαραγωγισιμότητα
random.seed(42)

patterns = [ "ATTAGA" , "ACGCATTT" , "AGGACTCAA" , "ATTTCAGT" ]

def mutate_pattern(pattern: str, max_mutations: int = 2) -> str:
    """
    Εφαρμόζει έως max_mutations τυχαίες μεταλλάξεις:
    - Κάθε μετάλλαξη: 50% substitution (σε ένα τυχαίο σύμβολο), 50% deletion.
    :param pattern: Αρχικό pattern
    :param max_mutations: Μέγιστος αριθμός μεταλλάξεων
    :return: Τροποποιημένο pattern
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
    Προσθέτει στο τέλος min_len–max_len τυχαία σύμβολα.
    :param min_len: Ελάχιστο μήκος της τυχαίας συμβολοσειράς
    :param max_len: Μέγιστο μήκος της τυχαίας συμβολοσειράς
    :return: Τυχαία συμβολοσειρά από τα σύμβολα A, C, G, T
    """
    k = random.randint(min_len, max_len)
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=k))

def save_fasta(seqs: List[str], filename: str) -> bool:
    """
    Αποθηκεύει τις αλληλουχίες σε αρχείο FASTA.
    :param seqs: Λίστα αλληλουχιών
    :param filename: Όνομα αρχείου
    :return: True αν η αποθήκευση ήταν επιτυχής, αλλιώς False
    """
    try:
        with open(filename, 'w') as f:
            for i, s in enumerate(seqs, 1):
                f.write(f">seq{i}\n{s}\n")
        return True
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
        return False

def save_txt(seqs: List[str], filename: str) -> bool:
    """
    Αποθηκεύει τις αλληλουχίες σε αρχείο txt.
    :param seqs: Λίστα αλληλουχιών
    :param filename: Όνομα αρχείου
    :return: True αν η αποθήκευση ήταν επιτυχής, αλλιώς False
    """
    try:
        with open(filename, 'w') as f:
            for i, s in enumerate(seqs, 1):
                f.write(f"seq{i}: {s}\n")
        return True
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
        return False


def generate_sequence ( ) -> str :
    """
    Δημιουργεί μία σύνθετη αλληλουχία σύμφωνα με:
    - 1–3 αργικά σύμβολα στην αρχή
    - Επικαλύψεις των patterns με έως 2 mutations το καθένα
    - 1–2 σύμβολα στο τέλος
    :return: Σύνθετη αλληλουχία
    """
    seq = create_random_sequence ( 1 , 3 )
    for pat in patterns :
        mut = mutate_pattern ( pat , max_mutations = 2 )
        seq += mut
    seq += create_random_sequence ( 1 , 2 )
    return seq

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

if save_fasta( datasetA, "auxiliary/datasetA.fasta" ):
    print("datasetA αποθηκεύτηκε σε datasetA.fasta")
if save_fasta( datasetB, "auxiliary/datasetB.fasta" ):
    print("datasetB αποθηκεύτηκε σε datasetB.fasta")
if save_fasta( datasetC, "auxiliary/datasetC.fasta" ):
    print("datasetC αποθηκεύτηκε σε datasetC.fasta")
if save_txt( datasetA, "auxiliary/datasetA.txt" ):
    print("datasetA αποθηκεύτηκε σε datasetA.txt")
if save_txt( datasetB, "auxiliary/datasetB.txt" ):
    print("datasetB αποθηκεύτηκε σε datasetB.txt")
if save_txt( datasetC, "auxiliary/datasetC.txt" ):
    print("datasetC αποθηκεύτηκε σε datasetC.txt")

# print statistics for each dataset
def print_statistics(dataset: List[str], name: str):
    print(f"Statistics for {name}:")
    print(f"  Number of sequences: {len(dataset)}")
    print(f"  Length of first sequence: {len(dataset[0])}")
    print(f"  Length of last sequence: {len(dataset[-1])}")
    print(f"  average length: {sum(len(seq) for seq in dataset) / len(dataset):.2f}")
    print(f"  First sequence: {dataset[0]}")
    print(f"  Last sequence: {dataset[-1]}")
    print()
print_statistics(datasetA, "datasetA")
print_statistics(datasetB, "datasetB")
print_statistics(datasetC, "datasetC")

