def load_rbns_data(rbns_filenames):
    rbns_data = []
    for filename in rbns_filenames:
        with open(filename, 'r') as file:
            for line in file:
                sequence, count = line.strip().split('\t')
                rbns_data.append((sequence, int(count)))
    return rbns_data

def load_rnacompete_probes(rnacompete_filename):
    rnacompete_probes = []
    with open(rnacompete_filename, 'r') as file:
        for line in file:
            sequence = line.strip()
            rnacompete_probes.append(sequence)
    return rnacompete_probes



# Example usage
rbns_filenames = ["rbns_data1.txt", "rbns_data2.txt", "rbns_data3.txt"]
rnacompete_filename = "rnacompete_probes.txt"

rbns_data = load_rbns_data(rbns_filenames)
rnacompete_probes = load_rnacompete_probes(rnacompete_filename)

## ---- 

def preprocess_rbns_data(rbns_data):
    # Sort the RBNS data lexicographically by sequence
    sorted_rbns_data = sorted(rbns_data, key=lambda x: x[0])
    return sorted_rbns_data

def preprocess_rnacompete_probes(rnacompete_probes, min_length, max_length):
    # Filter out invalid probe sequences based on length
    filtered_probes = [probe for probe in rnacompete_probes if min_length <= len(probe) <= max_length]
    return filtered_probes

# Example usage
sorted_rbns_data = preprocess_rbns_data(rbns_data)
filtered_rnacompete_probes = preprocess_rnacompete_probes(rnacompete_probes, min_length=30, max_length=41)

## -----
def extract_kmers(sequence, k):
    kmers = []
    sequence_length = len(sequence)
    for i in range(sequence_length - k + 1):
        kmer = sequence[i:i+k]
        kmers.append(kmer)
    return kmers

def extract_kmers_from_rbns(rbns_data, k):
    rbns_kmers = []
    for sequence, count in rbns_data:
        kmers = extract_kmers(sequence, k)
        rbns_kmers.extend(kmers)
    return rbns_kmers

def extract_kmers_from_rnacompete(rnacompete_probes, k):
    rnacompete_kmers = []
    for probe in rnacompete_probes:
        kmers = extract_kmers(probe, k)
        rnacompete_kmers.extend(kmers)
    return rnacompete_kmers

# Example usage
k = 4
rbns_kmers = extract_kmers_from_rbns(sorted_rbns_data, k)
rnacompete_kmers = extract_kmers_from_rnacompete(filtered_rnacompete_probes, k)

## -----
def compute_kmer_scores(rbns_kmers):
    kmer_scores = {}
    for kmer in rbns_kmers:
        if kmer in kmer_scores:
            kmer_scores[kmer] += 1
        else:
            kmer_scores[kmer] = 1
    return kmer_scores

# Example usage
rbns_kmer_scores = compute_kmer_scores(rbns_kmers)

## -----
def predict_binding_intensity(rnacompete_probes, kmer_scores, k):
    binding_intensities = []
    for probe in rnacompete_probes:
        probe_kmers = extract_kmers(probe, k)
        intensity = sum(kmer_scores.get(kmer, 0) for kmer in probe_kmers)
        binding_intensities.append(intensity)
    return binding_intensities

# Example usage
k = 4
rnacompete_binding_intensities = predict_binding_intensity(filtered_rnacompete_probes, rbns_kmer_scores, k)

## -----
def write_binding_intensities(output_filename, rnacompete_probes, binding_intensities):
    with open(output_filename, 'w') as file:
        for probe, intensity in zip(rnacompete_probes, binding_intensities):
            file.write(f"{probe}\t{intensity}\n")

# Example usage
output_filename = "predicted_binding_intensities.txt"
write_binding_intensities(output_filename, filtered_rnacompete_probes, rnacompete_binding_intensities)

## ----
def evaluate_accuracy(predicted_intensities, actual_intensities):
    total_probes = len(predicted_intensities)
    correct_predictions = sum(1 for pred, actual in zip(predicted_intensities, actual_intensities) if pred == actual)
    accuracy = correct_predictions / total_probes
    return accuracy

# Example usage
test_actual_intensities = [0.8, 0.9, 0.6, 0.4, 0.7]  # Example actual intensities from the test set
accuracy = evaluate_accuracy(rnacompete_binding_intensities, test_actual_intensities)
print(f"Accuracy: {accuracy}")