"""
2.	For a given set of training data examples stored in a .CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.
Origin	Manufacturer	Color	Decade	Type	Example Type
Japan	Honda	Blue	1980	Economy	Positive
Japan	Toyota	Green	1970	Sports	Negative
Japan	Toyota	Blue	1990	Economy	Positive
USA	Chrysler	Red	1980	Economy	Negative
Japan	Honda	White	1980	Economy	Positive

"""
import pandas as pd

# Define the dataset
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': [1980, 1970, 1990, 1980, 1980],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}

# Create a DataFrame
df = pd.DataFrame(data)

def more_general(h1, h2):
    """Checks if hypothesis h1 is more general than h2."""
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg_part = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg_part)
    return all(more_general_parts)

def min_generalizations(h, instance):
    """Returns a minimal generalization of hypothesis h to include instance."""
    h_new = list(h)
    for i in range(len(h)):
        if h_new[i] == "0":
            h_new[i] = instance[i]
        elif h_new[i] != instance[i]:
            h_new[i] = "?"
    return [tuple(h_new)]

def min_specializations(h, domains, instance):
    """Returns the minimal specializations of hypothesis h to exclude instance."""
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if instance[i] != val:
                    h_new = h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new = h[:i] + ("0",) + h[i+1:]
            results.append(h_new)
    return results

# Candidate-Elimination Algorithm
def candidate_elimination(df):
    domains = [set(df[col]) for col in df.columns[:-1]]
    S = {tuple(["0"] * (len(df.columns) - 1))}
    G = {tuple(["?"] * (len(df.columns) - 1))}
    
    for i, row in df.iterrows():
        instance = tuple(row.iloc[:-1])
        if row.iloc[-1] == 'Positive':
            # Remove all hypotheses from G that are inconsistent with the instance
            G = {g for g in G if more_general(g, instance)}
            # For each hypothesis s in S that is inconsistent with the instance
            new_S = set()
            for s in S:
                if not more_general(instance, s):
                    new_S.update(min_generalizations(s, instance))
                else:
                    new_S.add(s)
            # Keep only those generalizations that are more specific than some hypothesis in G
            S = {s for s in new_S if any(more_general(g, s) for g in G)}
        else:  # Negative example
            # Remove all hypotheses from S that are inconsistent with the instance
            S = {s for s in S if more_general(instance, s)}
            # For each hypothesis g in G that is inconsistent with the instance
            new_G = set()
            for g in G:
                if more_general(g, instance):
                    new_G.update(min_specializations(g, domains, instance))
                else:
                    new_G.add(g)
            # Keep only those specializations that are more general than some hypothesis in S
            G = {g for g in new_G if any(more_general(g, s) for s in S)}
    
    return S, G

# Run the Candidate-Elimination algorithm
S, G = candidate_elimination(df)
print("Most specific hypotheses in S:", S)
print("Most general hypotheses in G:", G)
