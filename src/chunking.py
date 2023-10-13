# Chunx

def chunk_random(data, frac, n):
    df = data.sample(frac=frac).reset_index(drop=True)
    l = len(df)
    data_subsets = [
        df.iloc[i*(l//n):(i+1)*(l//n)].reset_index(drop=True) for i in range(n)]
    return data_subsets


def chunk_segment(data, frac, segment_by, n):
    # reutrn a list of DATAFRAME chunks by a column (n most prevalent)
    return


def chunk_seed(data, frac, n):
    df = data.sample(frac=frac/n).reset_index(drop=True)
    data_subsets = [df for i in range(n)]
    return data_subsets

# TODO Bootstrap
