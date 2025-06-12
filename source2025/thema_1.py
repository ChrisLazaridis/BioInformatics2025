import random
from typing import List

# For reproducability

patterns = [ "ATTAGA" , "ACGCATTT" , "AGGACTCAA" , "ATTTCAGT" ]

def mutate_pattern(pattern: str, max_mutations: int = 2) -> str:
    """
    Μπορεί να αλλάξει το pattern με τυχαία αλλαγή χαρακτήρων ή αφαίρεση χαρακτήρων.
    Parameters:
        pattern (str): The pattern to mutate.
        max_mutations (int): The maximum number of mutations to perform.
    Returns:
        str: The mutated pattern.
    """
    pattern_list = list(pattern)
    m = random.randint(0, max_mutations)
    positions = random.sample(range(len(pattern_list)), m)
    for pos in positions:
        if random.random() < 0.5:
            pattern_list[pos] = random.choice(['A', 'C', 'G', 'T'])
        else:
            pattern_list[pos] = ''
    return ''.join(pattern_list)


def create_random_sequence(min_len: int = 1, max_len: int = 2) -> str:
    """
    Δημιουργεί μια τυχαία ακολουθία με μήκος μεταξύ min_len και max_len.
    Parameters:
        min_len (int): Μικρότερο δυνατό μήκος της ακολουθίας.
        max_len (int): Μέγιστο μήκος της ακολουθίας.
    Returns:
        str: The generated random sequence.
    """
    k = random.randint(min_len, max_len)
    return ''.join(random.choices(['A', 'C', 'G', 'T'], k=k))

def save_fasta(seqs: List[str], filename: str) -> bool:
    """
    Αποθηκεύει μια λίστα ακολουθιών σε αρχείο FASTA.
    Parameters:
        seqs (List[str]): Λίστα ακολουθιών.
        filename (str): Όνομα αρχείου για αποθήκευση.
    Returns:
        bool: True αν η αποθήκευση ήταν επιτυχής, αλλιώς False.
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
    Δημιουργεί μια τυχαία ακολουθία.
    Parameters:
        patterns (List[str]): Λίστα προτύπων.
    Returns:
        str: Η παραγόμενη ακολουθία.
    """
    seq = create_random_sequence ( 1 , 3 )
    for pat in patterns :
        mut = mutate_pattern ( pat , max_mutations = 2 )
        seq += mut
    seq += create_random_sequence ( 1 , 2 )
    return seq
def print_statistics(dataset: List[str], name: str):
    """
    Εκτυπώνει στατιστικά στοιχεία για μια λίστα ακολουθιών.
    Parameters:
        dataset (List[str]): Λίστα ακολουθιών.
        name (str): Όνομα του dataset.
    Returns:
        None
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

