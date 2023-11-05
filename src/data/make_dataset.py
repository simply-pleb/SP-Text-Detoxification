import pandas as pd

def swap_toxic_trn_with_ref(data):
    data_ref_tox = data.loc[data.ref_tox >= data.trn_tox]
    data_trn_tox = data.loc[data.ref_tox < data.trn_tox]
    
    data_trn_tox['reference'], data_trn_tox['translation'] = data_trn_tox['translation'], data_trn_tox['reference']
    data_trn_tox['ref_tox'], data_trn_tox['trn_tox'] = data_trn_tox['trn_tox'], data_trn_tox['ref_tox']

    data_vs = pd.concat([data_ref_tox, data_trn_tox])
    return data_vs

def choose_data(data):
    mean_similarity = data['similarity'].mean()
    mean_lenght_diff = data['lenght_diff'].mean()
    data_final = data.loc[(data.ref_tox >= 0.9) & (data.similarity >= mean_similarity) & (data.lenght_diff <= mean_lenght_diff)]

def change_paranmt_format_to_paradetox(data):
    data_intermit = data[['reference', 'translation']]
    data_intermit = data_intermit.rename(columns={'reference': 'toxic', 'translation': 'neutral1'})
    data_intermit['neutral2'] = pd.Series([None] * len(data_intermit))
    data_intermit['neutral3'] = pd.Series([None] * len(data_intermit))

    return data

def main():
    data = pd.read_csv("../data/raw/filtered.tsv", sep="\t")
    data = swap_toxic_trn_with_ref(data)
    data = choose_data(data)
    data = change_paranmt_format_to_paradetox(data)

    data_paradetox = pd.read_csv("../data/external/paradetox", sep="\t")
    data_result = pd.concat([data_paradetox, data], ignore_index=True)

    data_result.to_csv('../data/intermit/merged_dataset.tsv', sep='\t')


if __name__ == "__main__":
    main()