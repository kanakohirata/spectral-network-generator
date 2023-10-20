import pandas as pd


def create_compound_dict():
    return {'db_id': '', 'exact_mass': 0, 'inchi': '', 'inchikey': '', 'list_cas_rn': [],
            'list_cmpd_classification_superclass': [], 'list_cmpd_classification_class': [],
            'list_cmpd_classification_subclass': [], 'list_cmpd_classification_alternative_parent': [],
            'list_hmdb_id': [], 'molecular_formula': '', 'name': '', 'smiles': '', }


def read_multi_compound_table(paths, mode='dict'):
    """

    Parameters
    ----------
    paths: list
    mode: str
        'dict', 'list', or 'df'

    Returns
    -------
        if mode is 'dict':
            return {'inchikey0': {'property0': 'a0', 'property1': 'b0', 'property2': 'c0', ...},
                    'inchikey1': {'property0': 'a1', 'property1': 'b1', 'property2': 'c1', ...},
                    'inchikey2': {'property0': 'a2', 'property1': 'b2', 'property2': 'c2', ...}, ...}

        else if mode is 'list':
            return [{'inchikey': inchikey0', property0': 'a0', 'property1': 'b0', 'property2': 'c0', ...},
                    {'inchikey': inchikey1','property0': 'a1', 'property1': 'b1', 'property2': 'c1', ...},
                    {'inchikey': inchikey2','property0': 'a2', 'property1': 'b2', 'property2': 'c2', ...}, ...]
    """

    all_compounds = {}
    for path in paths:
        if path.endswith('.csv'):
            delimiter = ','
        elif path.endswith('.tsv'):
            delimiter = '\t'
        else:
            continue
        _compounds = read_compound_table(path, mode='dict', delimiter=delimiter)

        if not all_compounds:
            all_compounds = _compounds
        else:
            for _inchikey, _info in _compounds.items():
                if _inchikey in all_compounds:
                    all_compounds[_inchikey]['list_cas_rn'] = list(set(
                        all_compounds[_inchikey]['list_cas_rn'] + _info['list_cas_rn']))

                    all_compounds[_inchikey]['list_hmdb_id'] = list(set(
                        all_compounds[_inchikey]['list_hmdb_id'] + _info['list_hmdb_id']))

                    all_compounds[_inchikey]['list_cmpd_classification_superclass'] = list(set(
                        all_compounds[_inchikey]['list_cmpd_classification_superclass']
                        + _info['list_cmpd_classification_superclass']))

                    all_compounds[_inchikey]['list_cmpd_classification_class'] = list(set(
                        all_compounds[_inchikey]['list_cmpd_classification_class']
                        + _info['list_cmpd_classification_class']))

                    all_compounds[_inchikey]['list_cmpd_classification_subclass'] = list(set(
                        all_compounds[_inchikey]['list_cmpd_classification_subclass']
                        + _info['list_cmpd_classification_subclass']))
                else:
                    all_compounds[_inchikey] = _info

    if mode == 'dict':
        return all_compounds
    elif mode == 'list':
        return list(all_compounds.values())
    elif mode == 'df':
        return pd.DataFrame(all_compounds.values())


def read_compound_table(path, mode='dict', delimiter='\t'):
    """

    Parameters
    ----------
    path: str
    mode: str
        'dict', 'list', or 'df'
    delimiter: str
        '\t' or ','

    Returns
    -------
        if mode is 'dict':
            return {'inchikey0': {'property0': 'a0', 'property1': 'b0', 'property2': 'c0', ...},
                    'inchikey1': {'property0': 'a1', 'property1': 'b1', 'property2': 'c1', ...},
                    'inchikey2': {'property0': 'a2', 'property1': 'b2', 'property2': 'c2', ...}, ...}

        else if mode is 'list':
            return [{'inchikey': inchikey0', property0': 'a0', 'property1': 'b0', 'property2': 'c0', ...},
                    {'inchikey': inchikey1','property0': 'a1', 'property1': 'b1', 'property2': 'c1', ...},
                    {'inchikey': inchikey2','property0': 'a2', 'property1': 'b2', 'property2': 'c2', ...}, ...]
    """
    df = pd.read_csv(path, delimiter=delimiter, index_col=None, on_bad_lines='skip')

    columns_to_rename = {}
    columns_to_use = []
    for column_name in df.columns:
        if column_name in ['CAS RN', 'CAS rn', 'CAS_rn', 'CAS no', 'CAS NO']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'list_cas_rn'
            columns_to_use.append('list_cas_rn')
        elif column_name in ['DBID', 'DB_ID', 'DB ID']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'db_id'
            columns_to_use.append('db_id')
        elif column_name in ['HMDBID', 'HMDB_ID', 'HMDB ID']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'list_hmdb_id'
            columns_to_use.append('list_hmdb_id')
        elif column_name in ['InChI', 'inchi', 'INCHI', 'InChi']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'inchi'
            columns_to_use.append('inchi')
        elif column_name in ['InChIKey', 'inchikey', 'inchi_key', 'INCHIKEY', 'INCHI_KEY']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'inchikey'
            columns_to_use.append('inchikey')
        elif column_name in ['molecular formula', 'molecular_formula', 'MOLECULAR FORMULA', 'MOLECULAR_FORMULA', 'formula',
                             'FORMULA', 'elemental composition', 'elemental composition', 'ELEMENTAL COMPOSITION',
                             'ELEMENTAL_COMPOSITION']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'formula'
            columns_to_use.append('formula')
        elif column_name in ['monoisotopic mass', 'monoisotopic_mass', 'MONOISOTOPIC_MASS', 'MONOISOTOPIC MASS',
                             'exact mass', 'exact_mass', 'EXACT_MASS', 'EXACT MASS']:
            df[column_name].astype(float)
            df[column_name].fillna(0.0, inplace=True)
            columns_to_rename[column_name] = 'exact_mass'
            columns_to_use.append('exact_mass')
        elif column_name in ['NAME', 'name', 'COMMON_NAME', 'common name']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'name'
            columns_to_use.append('name')
        # chemical taxonomy
        elif column_name in ['Classyfire_superclass', 'superclass', 'SUPERCLASS']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'list_cmpd_classification_superclass'
            columns_to_use.append('list_cmpd_classification_superclass')
        elif column_name in ['Classyfire_class', 'class', 'CLASS']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'list_cmpd_classification_class'
            columns_to_use.append('list_cmpd_classification_class')
        elif column_name in ['Classyfire_subclass', 'subclass', 'SUBCLASS']:
            df[column_name].astype(str)
            df[column_name].fillna('', inplace=True)
            columns_to_rename[column_name] = 'list_cmpd_classification_subclass'
            columns_to_use.append('list_cmpd_classification_subclass')

    if ('inchikey' not in columns_to_use
            or 'inchi'not in columns_to_use
            or 'exact_mass' not in columns_to_use
            or 'formula' not in columns_to_use):

        raise ValueError(f'Column name is not correct: {path}\n'
                         f'\tcolumns: {df.columns}')

    df.rename(columns=columns_to_rename, inplace=True)
    df = df[columns_to_use]
    
    if 'list_hmdb_id' in columns_to_use:
        df['list_hmdb_id'] = df['list_hmdb_id'].str.split(',').apply(lambda x: [item.strip() for item in x if item.strip()])
    if 'list_cas_rn' in columns_to_use:
        df['list_cas_rn'] = df['list_cas_rn'].str.split(',').apply(lambda x: [item.strip() for item in x if item.strip()])
    if 'list_cmpd_classification_superclass' in columns_to_use:
        df['list_cmpd_classification_superclass']\
            = df['list_cmpd_classification_superclass'].apply(lambda x: [x] if x else [])
    if 'list_cmpd_classification_class' in columns_to_use:
        df['list_cmpd_classification_class']\
            = df['list_cmpd_classification_class'].apply(lambda x: [x] if x else [])
    if 'list_cmpd_classification_subclass' in columns_to_use:
        df['list_cmpd_classification_subclass']\
            = df['list_cmpd_classification_subclass'].apply(lambda x: [x] if x else [])

    df.drop_duplicates(subset='inchikey', inplace=True)

    if mode == 'list':
        return df.to_dict(orient='records')
    elif mode == 'dict':
        df.set_index('inchikey', inplace=True, drop=False)
        return df.to_dict(orient='index')
    elif mode == 'df':
        return df
    