import random

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict

import torch
from rdkit import Chem
from rdkit.Chem import  BRICS
import networkx as nx
from torch_geometric.data import HeteroData

from utils import *


# normalize the dictionary
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# protein dictionary
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
file_ide = 'data/RotatE_128_64.pkl'
with open(file_ide, 'rb') as f:
    embeddings = pickle.load(f)
id2entity_data = embeddings['id2entity']
id2relation_data=embeddings['id2relation']
data_str = ""
for id_, entity in id2entity_data.items():
    data_str += f"ID: {id_}, 实体: {entity}\n"
def build_entity_dict(data):
    entity_dict = {}
    for line in data.splitlines():
        if "ID:" in line and "实体:" in line:
            parts = line.split(", 实体: ")
            entity_id = int(parts[0].replace("ID: ", "").strip())
            entity_name = parts[1].strip()
            entity_dict[entity_name] = entity_id
    return entity_dict
entity2id= build_entity_dict(data_str)
data_str2 = ""
for id_, entity in id2relation_data.items():
    data_str2 += f"ID: {id_}, 实体: {entity}\n"
relation2id= build_entity_dict(data_str2)
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
def parse_triples(file_path):
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            entity1 = parts[0]
            id1 = entity2id[entity1]
            relation = parts[1]
            id2 = relation2id[relation]
            entity2 = parts[2]
            id3=entity2id[entity2]
            triples.append((id1,id2,id3))
    return triples
file_path = 'data/triples.txt'
triples = parse_triples(file_path)
# target graph node features
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# get residue features
def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# drug sequence dictionary------------------------------------------------------
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


# map the symbol of atom to label
def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


# get all drug graph node features ------------------------------------------------------------------------
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])
file_path = 'data/RotatE_128_64_emb.pkl'
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    entity_embeddings = embeddings['entity_emb']
    relation_embeddings=embeddings['relation_emb']
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_attribute_features(entity_name):
    if entity_name in entity2id:
        entity_id = entity2id[entity_name]
        attribute_embedding = entity_embeddings[entity_id]
    else:
        raise ValueError(f"实体 '{entity_name}' 不存在于 relation2id 映射中，程序中断。")
    return np.array(attribute_embedding)
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    atom_index=[]

    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol in entity2id:
            atom_id = entity2id[atom_symbol]
        atom_index.append(atom_id)
        feature = get_atom_features(atom)
        features.append(feature)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    attribute_features = []
    attribute_edge_index = [[], []]
    attribute_edge_index2 = [[], []]
    relation_features = []
    attribute_to_idx = {}
    next_attribute_idx = 0
    id=0
    for atom_id in atom_index:
        for (attribute, relation, atom_entity) in triples:
            if atom_entity == atom_id:
                if attribute not in attribute_to_idx:
                    attribute_idx = next_attribute_idx  # 确保属性索引从原子数量之后开始
                    attribute_to_idx[attribute] = attribute_idx  # 存储该属性的索引
                    attribute_embedding = np.array(entity_embeddings[attribute])  # 获取属性嵌入
                    attribute_features.append(attribute_embedding)
                    next_attribute_idx += 1  # 增加属性索引
                else:
                    attribute_idx = attribute_to_idx[attribute]
                relation_embedding = np.array(relation_embeddings[relation])
                attribute_edge_index[0].append(attribute_idx)
                attribute_edge_index[1].append(id)
                attribute_edge_index2[1].append(attribute_idx)
                attribute_edge_index2[0].append(id)
                relation_features.append(relation_embedding)
        id+=1
    for atom in mol.GetAtoms():
        atom.SetIntProp('molAtomIdx', atom.GetIdx())
    brics_bonds = BRICS.FindBRICSBonds(mol)
    bond_indices = [mol.GetBondBetweenAtoms(pair[0][0], pair[0][1]).GetIdx() for pair in brics_bonds]
    if not bond_indices:
        num_atoms = mol.GetNumAtoms()
        fragment_features = [np.zeros(128)]
        atom_to_fragment_edges = [list(range(num_atoms)), [0] * num_atoms]
        frag_edge_index = [[0], [0]]
    else:
        mol_frag = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)
        frags = Chem.GetMolFrags(mol_frag, asMols=False, sanitizeFrags=False)
        atom_to_fragment = {}
        for frag_idx, atom_indices in enumerate(frags):
            for atom_idx in atom_indices:
                atom = mol_frag.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() != '*':
                    mol_atom_idx = atom.GetIntProp('molAtomIdx')
                    atom_to_fragment[mol_atom_idx] = frag_idx
        num_atoms = mol.GetNumAtoms()
        num_fragments = len(frags)
        fragment_features = [np.zeros(128) for _ in range(num_fragments)]
        atom_to_fragment_edges = [[], []]
        for atom_idx in range(num_atoms):
            if atom_idx in atom_to_fragment:
                frag_idx = atom_to_fragment[atom_idx]
                atom_to_fragment_edges[0].append(atom_idx)
                atom_to_fragment_edges[1].append(frag_idx)  # 分子片段索引从 0 开始
            else:
                raise ValueError(f"原子索引 {atom_idx} 未找到对应的分子片段。")

        # 构建分子片段之间的边，不调整分子片段节点的索引
        fragment_edge_set = set()
        for bond in mol.GetBonds():
            atomA_idx = bond.GetBeginAtomIdx()
            atomB_idx = bond.GetEndAtomIdx()
            fragA_idx = atom_to_fragment.get(atomA_idx)
            fragB_idx = atom_to_fragment.get(atomB_idx)
            if fragA_idx is not None and fragB_idx is not None and fragA_idx != fragB_idx:
                fragment_edge_set.add((fragA_idx, fragB_idx))
                fragment_edge_set.add((fragB_idx, fragA_idx))  # 添加双向边
        frag_edge_index = []
        for u, v in fragment_edge_set:
            frag_edge_index.append([u, v])
    return c_size, features, edge_index,attribute_features,attribute_edge_index,relation_features,fragment_features,atom_to_fragment_edges,frag_edge_index


def get_atom_features(atom):
    atom_symbol = atom.GetSymbol()
    if atom_symbol in entity2id:
        atom_id = entity2id[atom_symbol]
        atom_embedding = entity_embeddings[atom_id]
    else:
        raise ValueError(f"原子符号 '{atom_symbol}' 不存在于 entity2id 映射中，程序中断。")

    return np.array(atom_embedding)

# map target sequence to label--------------------------------------------------------
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


# target graph node fature (pssm)---------------------------------------------------------
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue

                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    # k = float(len(pro_res_table))
    # pwm_mat = np.log2(ppm_mat / (1.0 / k))
    # pssm_mat = pwm_mat
    # print(pssm_mat)
    return pssm_mat


# get all target graph node features except pssm
def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


# concatenate pssm and other node features
def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    # print('target_feature')
    # print(pssm.shape)
    # print(other_feature.shape)

    # print(other_feature.shape)
    # return other_feature
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


# get all target graph node feature
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence)
    return feature


def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0], dtype=np.int64))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)

    return target_size, target_feature, target_edge_index


def valid_target(key, dataset):
    contact_dir = 'data/' + dataset + '/contact_map'
    aln_dir = 'data/' + dataset + '/aln'
    contact_file = os.path.join(contact_dir, key + '.npy')
    aln_file = os.path.join(aln_dir, key + '.aln')
    if os.path.exists(contact_file) and os.path.exists(aln_file):
        return True
    else:
        return False


# -create drug sequence, drug graph, target sequence , target graph


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def create_datas3(dataset, fold=0):
    # ------------------------------create  CSV file-------------------------------------------
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    fold = train_fold + valid_fold
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/contact_map'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # get smiles sequence list
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)  # loading drugs
        drug_smiles.append(ligands[d])
        # get target sequence list
    for t in proteins.keys():
        prots.append(proteins[t])  # loading proteins
        prot_keys.append(t)

    if dataset == 'davis' or 'filter_davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)  # affinity shape=(68 drug,442 prot)
    opts = ['train', 'test']
    m = affinity.shape[1]
    arr = list(range(m))
    random.shuffle(arr)
    split_point = int(len(arr) * 5 / 6)
    tar = arr[:split_point]
    retar = arr[split_point:]
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
        rows, cols = rows[fold], cols[
            fold]
        result = np.column_stack((rows, cols))
        if opt == 'train':
            result = result[np.isin(result[:, 1], tar)]
            rows, cols = result[:, 0] , result[:, 1]
        elif opt == 'test':
            result = result[np.isin(result[:, 1], retar)]
            rows, cols = result[:, 0] , result[:, 1]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_key,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):  # Check if there are aln and pconsc4 files
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')  # csv format
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)):', len(set(drugs)), '---len(set(prots)):', len(set(prots)))
    # all_prots += list(set(prots))
    print('finish', dataset, ' csv file')

    compound_iso_smiles = drugs
    target_key = prot_keys
    # durg graph data
    smile_graph = {}
    smile_mtgraph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print('finish drug graph')
    # drug sequence data
    smile_tensor = {}
    smile=compound_iso_smiles
    pro=prots
    for smile in compound_iso_smiles:
        smi_tensor = label_smiles(smile, 100, CHARISOSMISET)
        smile_tensor[smile] = smi_tensor
    print('finish drug sequence')
    # target graph data
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g
    print('finish target graph')

    if len(target_graph) == 0:
        raise Exception('没有 aln 文件和 contact_map文件。')
    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    # target sequence data
    train_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in train_prots]

    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    test_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in test_prots]

    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
    print('ready for train_data and test_data')
    train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots,
                                y=train_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor,
                                target_graph=target_graph, target_key=train_prot_keys,smile=smile)
    test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,
                               y=test_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor, target_graph=target_graph,
                               target_key=test_prot_keys,smile=smile)
    print('finish train_data and test_data')

    return train_data, test_data

def create_dataset(dataset, fold=0):
    # ------------------------------create  CSV file-------------------------------------------
    fpath = 'data/' + dataset + '/'
    fpath2='data/filter davis/'
    train_fold = json.load(open(fpath2 + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath2 + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/contact_map'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)  # loading drugs
        drug_smiles.append(ligands[d])

    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis' or 'filter_davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'test']

    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[
                train_fold]
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_key,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('finish', dataset, ' csv file')

    compound_iso_smiles = drugs
    target_key = prot_keys
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print('finish drug graph')
    smile_tensor = {}
    smile=compound_iso_smiles
    pro=prots
    for smile in compound_iso_smiles:
        smi_tensor = label_smiles(smile, 100, CHARISOSMISET)
        smile_tensor[smile] = smi_tensor
    print('finish drug sequence')
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g
    print('finish target graph')

    if len(target_graph) == 0:
        raise Exception('没有 aln 文件和 contact_map文件。')
    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    train_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in train_prots]

    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    test_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in test_prots]

    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
    print('ready for train_data and test_data')
    train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots,
                                y=train_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor,
                                target_graph=target_graph, target_key=train_prot_keys,smile=smile)
    test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,
                               y=test_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor, target_graph=target_graph,
                               target_key=test_prot_keys,smile=smile)
    print('finish train_data and test_data')

    return train_data, test_data

def create_datas2(dataset, fold=0):
    # ------------------------------create  CSV file-------------------------------------------
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    fold = train_fold + valid_fold
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/contact_map'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # get smiles sequence list
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)  # loading drugs
        drug_smiles.append(ligands[d])
        # get target sequence list
    for t in proteins.keys():
        prots.append(proteins[t])  # loading proteins
        prot_keys.append(t)

    if dataset == 'davis' or 'filter_davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']
    m = affinity.shape[0]
    arr = list(range(m))
    random.shuffle(arr)
    split_point = int(len(arr) * 5 / 6)
    tar = arr[:split_point]
    retar = arr[split_point:]
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
        rows, cols = rows[fold], cols[
            fold]
        result = np.column_stack((rows, cols))
        if opt == 'train':
            result = result[np.isin(result[:, 0], tar)]
            rows, cols = result[:, 0] , result[:, 1]
        elif opt == 'test':
            result = result[np.isin(result[:, 0], retar)]
            rows, cols = result[:, 0] , result[:, 1]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_key,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):  # Check if there are aln and pconsc4 files
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')  # csv format
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)):', len(set(drugs)), '---len(set(prots)):', len(set(prots)))
    # all_prots += list(set(prots))
    print('finish', dataset, ' csv file')

    compound_iso_smiles = drugs
    target_key = prot_keys
    # durg graph data
    smile_graph = {}
    smile_mtgraph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print('finish drug graph')
    # drug sequence data
    smile_tensor = {}
    smile=compound_iso_smiles
    pro=prots
    for smile in compound_iso_smiles:
        smi_tensor = label_smiles(smile, 100, CHARISOSMISET)
        smile_tensor[smile] = smi_tensor
    print('finish drug sequence')
    # target graph data
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g
    print('finish target graph')

    if len(target_graph) == 0:
        raise Exception('没有 aln 文件和 contact_map文件。')
    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    # target sequence data
    train_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in train_prots]

    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    test_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in test_prots]

    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
    print('ready for train_data and test_data')
    train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots,
                                y=train_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor,
                                target_graph=target_graph, target_key=train_prot_keys,smile=smile)
    test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,
                               y=test_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor, target_graph=target_graph,
                               target_key=test_prot_keys,smile=smile)
    print('finish train_data and test_data')

    return train_data, test_data

def create_datas4(dataset, fold=0):
    # ------------------------------create  CSV file-------------------------------------------
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    fold = train_fold + valid_fold
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/contact_map'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # get smiles sequence list
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)  # loading drugs
        drug_smiles.append(ligands[d])
        # get target sequence list
    for t in proteins.keys():
        prots.append(proteins[t])  # loading proteins
        prot_keys.append(t)

    if dataset == 'davis' or 'filter_davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)  # affinity shape=(68 drug,442 prot)
    opts = ['train', 'test']
    m = affinity.shape[1]
    arr = list(range(m))
    random.shuffle(arr)
    split_point = int(len(arr) *0.9)
    tar = arr[:split_point]
    retar = arr[split_point:]
    h = affinity.shape[0]
    arr1 = list(range(h))
    random.shuffle(arr1)
    split_point1 = int(len(arr1) * 0.9)
    tar1 = arr[:split_point1]
    retar1 = arr[split_point1:]
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
        rows, cols = rows[fold], cols[
            fold]
        result = np.column_stack((rows, cols))
        if opt == 'train':
            result=result[np.isin(result[:, 0], tar1) & np.isin(result[:, 1], tar)]
            rows, cols = result[:, 0] , result[:, 1]
        elif opt == 'test':
            result=result[np.isin(result[:, 0], retar1) & np.isin(result[:, 1], retar)]
            rows, cols = result[:, 0] , result[:, 1]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_key,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):  # Check if there are aln and pconsc4 files
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')  # csv format
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)):', len(set(drugs)), '---len(set(prots)):', len(set(prots)))
    # all_prots += list(set(prots))
    print('finish', dataset, ' csv file')

    compound_iso_smiles = drugs
    target_key = prot_keys
    # durg graph data
    smile_graph = {}
    smile_mtgraph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print('finish drug graph')
    # drug sequence data
    smile_tensor = {}
    smile=compound_iso_smiles
    pro=prots
    for smile in compound_iso_smiles:
        smi_tensor = label_smiles(smile, 100, CHARISOSMISET)
        smile_tensor[smile] = smi_tensor
    print('finish drug sequence')
    # target graph data
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g
    print('finish target graph')

    if len(target_graph) == 0:
        raise Exception('没有 aln 文件和 contact_map文件。')
    df = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
        df['affinity'])
    # target sequence data
    train_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in train_prots]

    train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

    df = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    test_prot_keys = np.asarray(list(df['target_key']))
    XT = [seq_cat(t) for t in test_prots]

    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
    print('ready for train_data and test_data')
    train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots,
                                y=train_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor,
                                target_graph=target_graph, target_key=train_prot_keys,smile=smile)
    test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,
                               y=test_Y, smile_graph=smile_graph,smile_mtgraph=None, smile_tensor=smile_tensor, target_graph=target_graph,
                               target_key=test_prot_keys,smile=smile)
    print('finish train_data and test_data')

    return train_data, test_data