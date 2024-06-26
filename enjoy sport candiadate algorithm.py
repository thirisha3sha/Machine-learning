"""For a given set of training data examples stored in a .CSV file,
implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.

Example	Sky	Air Temp	Humidity	Wind	Water	Forecast	Enjoy Sport
1	Sunny	Warm	Normal	Strong	Warm	Same	Yes
2	Sunny	Warm	High	Strong	Warm	Same	Yes
3	Rainy	Cold	High	Strong	Warm	Change	No
4	Sunny	Warm	High	Strong	Cool	Change	Yes
"""
import pandas as pd
data = {
    'Example': [1, 2, 3, 4],
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}
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
                if instance[i]!= val:
                    h_new = h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new = h[:i] + ("0",) + h[i+1:]
            results.append(h_new)
    return results
def candidate_elimination(df):
    domains = [set(df[col]) for col in df.columns[:-1]]
    S = {tuple(["0"] * (len(df.columns) - 1))}
    G = {tuple(["?"] * (len(df.columns) - 1))}
    
    for i, row in df.iterrows():
        instance = tuple(row.iloc[:-1])
        if row.iloc[-1] == 'Yes':
            G = {g for g in G if more_general(g, instance)}
            new_S = set()
            for s in S:
                if not more_general(instance, s):
                    new_S.update(min_generalizations(s, instance))
                else:
                    new_S.add(s)
            S = {s for s in new_S if any(more_general(g, s) for g in G)}
        else:
            S = {s for s in S if more_general(instance, s)}
            new_G = set()
            for g in G:
                if more_general(g, instance):
                    new_G.update(min_specializations(g, domains, instance))
                else:
                    new_G.add(g)
            G = {g for g in new_G if any(more_general(g, s) for s in S)}
    
    return S, G
S, G = candidate_elimination(df)
print("Most specific hypotheses in S:", S)
print("Most general hypotheses in G:", G)
