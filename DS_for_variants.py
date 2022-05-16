"""
All the scripts needed to run DeepSea on the variants. 
Save each prediction in the specified directory. 
"""

import sys
import time
from deepsea_expecto_convert import *
from selene_sdk.utils import load_model_from_state_dict
import numpy as np
import selene_sdk
try:
    import torch
except:
    print('error importing torch')
import warnings
import math
from typing import Dict, Tuple, List
import pandas as pd
import h5py
import os


# dictionary which was used to train DeepSea
BASE_DICT = {
    'A': [1, 0, 0, 0],
    'G': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'a': [1, 0, 0, 0],
    'c': [0, 0, 1, 0],
    'g': [0, 1, 0, 0],
    't': [0, 0, 0, 1],
    'n': [0, 0, 0, 0],
    'N': [0, 0, 0, 0],
    '-': [0, 0, 0, 0]
}

DS_WIDTH = 2000  # width of the window that is used

genome = selene_sdk.sequences.Genome('resources/reference_files/hg19.fa')


def encodeSeqs(seqs, inputsize: int = 2000, concat_forward_reverse: bool = False) -> np.ndarray:
    """Convert sequences to 0-1 encoding according to BASE_DICT and truncate to the input size.
    If concat_forward_reverse: output concatenates the forward and reverse 
    complement sequence encodings.
    Args:
        seqs: list of sequences
        inputsize: the number of basepairs to encode in the output
        concat_forward_reverse: whether to concatenate revrse sequence
    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize
    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0)))                     :int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = BASE_DICT[c]
        n = n + 1

    if concat_forward_reverse:
        dataflip = seqsnp[:, ::-1, ::-1]
        seqsnp = np.concatenate([seqsnp, dataflip], axis=0)

    return seqsnp.astype("float32")

def check_if_reference(allele, location, chrom) -> bool:
    base_genome = genome.get_sequence_from_coords(
            chrom=chrom, start=location-1, end=location, strand='+')
    if base_genome.lower()==allele.lower():
        return True
    return False

def check_pulled_seq_pass(sequence: np.ndarray, allele1: str, allele2: str, center: int) -> Tuple[bool, bool]:
    """
    NOT USED YET
    Check pulled sequences: 
        - non-empty sequence
        - which allele was reference
    Return: 
        Bool1: sequence is valid
        Bool2: False is allele1 is reference allele 
    """
    if sequence.shape[0] == 0:  # empty
        return False, False

    if not np.array_equal(sequence[center-1], BASE_DICT[allele1]):
        # allele 1 is not reference
        if not np.array_equal(sequence[center-1], BASE_DICT[allele2]):
            # allele2 is also not reference
            return False, False
        else:
            return True, True
    return True, False


def mutate_sequence(sequence: str, start_all: int, end_all: int, mutation_pos: int,
                    new_base: str, strand: str, var_id: str = '') -> str:
    """
    Mutate sequence at the specified position, no reference value. 
    

    Input is a sequence as a string, and output is mutated sequence, also as a string.
    If new_base IS reference, return '' (empty string)
    ASSUMPTION: the mutation ref/alt is assumed to be w.r.t. '+' strand.

    Args:
        sequence: sequence as string
        start_all: start location from where the whole sequence was pulled
        end_all: end location from where the whole sequence is pulled
        mutation_pos: position of mutation on the chromosome (NOT relative)
        new_base: string, new base to use
        strand: '-' or '+', strand from which sequence was pulled
        var_id: optional, variant ID for debugging purposes 
    """
    # mutation dictionary
    ref_alt = {'a': 't', 'g': 'c', 'c': 'g', 't': 'a', 'A': 'T',
               'G': 'C', 'C': 'G', 'T': 'A', 'N': 'N', 'n': 'n'}
    # note that behavior is different depending on the strand
    # if negative strand use the "end" location
    if strand == '-':
        try:
            base_at_loc = sequence[end_all-mutation_pos]
        except:
            print('could not access mutation', start_all, end_all, mutation_pos)
            return ''

        # reference from file and sequence reference are not matched - error
        if (ref_alt[base_at_loc.lower()] == new_base.lower()):
            # we have a reference (no mutation needed)
            return ''
        return sequence[:(end_all-mutation_pos)]+ref_alt[new_base]+sequence[(end_all-mutation_pos+1):]
    else:
        try:
            base_at_loc = sequence[mutation_pos-start_all-1]
        except:
            print('could not access mutation', start_all, end_all, mutation_pos)
            return ''
        if base_at_loc.lower() == new_base.lower():
            # we have a reference (no mutation needed)
            return ''
        return sequence[:(mutation_pos-start_all-1)]+new_base+sequence[(mutation_pos-start_all):]


def mutate_sequence_with_ref(sequence: str, start_all: int, end_all: int, mutation_pos: int,
                    ref: str, alt: str, strand: str, var_id: str = '') -> str:
    """
    Mutate sequence at the specified position.

    Input is a sequence as a string, and output is mutated sequence, also as a string.
    ASSUMPTION: the mutation ref/alt is assumed to be w.r.t. '+' strand.

    Args:
        sequence: sequence as string
        start_all: start location from where the whole sequence was pulled
        end_all: end location from where the whole sequence is pulled
        mutation_pos: position of mutation on the chromosome (NOT relative)
        ref: string, base at the reference
        alt: string, new base that should be placed
        strand: '-' or '+', strand from which sequence was pulled
        var_id: optional, variant ID for debugging purposes 
    """
    # mutation dictionary
    ref_alt = {'a': 't', 'g': 'c', 'c': 'g', 't': 'a', 'A': 'T',
               'G': 'C', 'C': 'G', 'T': 'A', 'N': 'N', 'n': 'n'}
    # note that behavior is different depending on the strand
    # if negative strand use the "end" location
    if strand == '-':
        try:
            base_at_loc = sequence[end_all-mutation_pos]
        except:
            print('could not access mutation', end_all, mutation_pos)
            return ''

        # reference from file and sequence reference are not matched - error
        if (ref_alt[base_at_loc.lower()] != ref.lower()):
            print('MISMATCH')
            print('- strand')
            print(start_all, mutation_pos)
            print(var_id)
            print(base_at_loc.lower(), ref.lower())
            print(sequence[end_all-mutation_pos-3:end_all-mutation_pos+3], ref)
            raise ValueError(
                'Mismatch in reference pulled and reference in dataframe')
        return sequence[:(end_all-mutation_pos)]+ref_alt[alt]+sequence[(end_all-mutation_pos+1):]
    else:
        try:
            base_at_loc = sequence[mutation_pos-start_all-1]
        except:
            print('could not access mutation', start_all, mutation_pos)
            return ''
        if base_at_loc.lower() != ref.lower():
            # Error message for debugging
            print('MISMATCH')
            print(start_all, mutation_pos)
            print(var_id)
            print(strand)
            print(base_at_loc.lower(), ref.lower())
            print(sequence[mutation_pos-start_all -
                           3:mutation_pos-start_all+3], ref)
            raise ValueError(
                'Mismatch in reference pulled and reference in dataframe')
        return sequence[:(mutation_pos-start_all-1)]+alt+sequence[(mutation_pos-start_all):]


def check_alignment(mutation_positions, genome):
    """Function to check reference vs alternative values. Raises error if mismatch found. 

    mutation_positions: needs to have fields chromosome, ref, pos
    Assumes that reference is w/ respect to the + strand

    """
    for i, mut in mutation_positions.iterrows():
        base_genome = genome.get_sequence_from_coords(
            chrom=mut.chrom, start=mut.snp_chromStart, end=mut.snp_chromEnd, strand='+')
        if base_genome != mut.ref:
            print(base_genome)
            print(mut)
            raise ValueError(
                'Mismatch in the mutation file and reference genome:')


def get_full_model_prediction(*, model, chrom: str, strand: str, center: int, mutation_pos: int = -1, ref: str = '',
                              alt: str = '', var_id: str = '', run_ref: bool = True, run_alt: bool = True, two_way_avg: bool = True,
                              use_cuda: bool = False):
    """ Function to get model predictions for the sequence with or without mutation @ specified position.

    If ref is provided, check base @ reference position

    Args:
        model: model to get predictions
        chrom: chrN format, chromosomal location of the sequence
        strand: '+' or '-' strand
        mutation_pos: absolute value of the mutation position on the chromosome
        ref: base @ reference position
        alt: base that we mutate to
        var_id: (optional) variant ID for the mutaiton
        run_ref: whether to evaluate reference (to avoid repeated evaluation)
        run_alt: whether to mutate sequence
        two_way_avg: whether to average DeepSea outputs 2 way (x2 longer runtime)
        use_cuda: is CUDA available
    """

    debugging_starts = []

    # define shift used for sliding window
    # range does not include last value
    shifts = np.array(list(range(-20000, 20200, 200)))

    # start and end for the whole window (inside which we do sliding window)
    start_all = int(center - int(0.5*DS_WIDTH) + shifts[0])
    end_all = int(center + int(0.5*DS_WIDTH) + shifts[-1])

    # pull the whole sequence at once
    seq_whole_ref = genome.get_sequence_from_coords(
        chrom=chrom, start=start_all, end=end_all, strand=strand)

    # mutate if needed
    if run_alt:
        if ref:
            seq_mutated = mutate_sequence_with_ref(
                seq_whole_ref, start_all, end_all, mutation_pos, ref, alt, strand)
        else:
            seq_mutated = mutate_sequence(
                seq_whole_ref, start_all, end_all, mutation_pos, alt, strand)
            if not seq_mutated:
                run_alt = False

    # define sequences we will use and output shapes
    if run_ref and run_alt:
        sequences_run = [seq_whole_ref, seq_mutated]
        output_gene = np.zeros([2, 2002, 201])  # reference and alternative
    elif run_alt:
        sequences_run = [seq_mutated]
        output_gene = np.zeros([1, 2002, 201])  # only alternative
    elif run_ref:
        sequences_run = [seq_whole_ref]
        output_gene = np.zeros([1, 2002, 201])  # only reference
    else:
        return None, None

    # implement sliding window
    for which_seq, seq_whole in enumerate(sequences_run):
        count = 0
        for s in shifts:
            start_idx = 20000+s
            end_idx = 20000+s+DS_WIDTH
            debugging_starts.append(start_idx)
            seq = seq_whole[start_idx:end_idx]
            seq = encodeSeqs([seq], concat_forward_reverse=two_way_avg)

            # average over forward and reverse to smooth noise if two_way_avg
            if two_way_avg:
                if use_cuda:
                    out_fw = model.forward(torch.tensor(
                        seq[0]).cuda().unsqueeze(0).unsqueeze(2))
                    our_rv = model.forward(torch.tensor(
                        seq[1]).cuda().unsqueeze(0).unsqueeze(2))
                    out = (out_fw+our_rv)/2
                else:
                    out_fw = model.forward(torch.tensor(
                        seq[0]).unsqueeze(0).unsqueeze(2))
                    our_rv = model.forward(torch.tensor(
                        seq[1]).unsqueeze(0).unsqueeze(2))
                    out = (out_fw+our_rv)/2
            else:
                if use_cuda:
                    out = model.forward(torch.tensor(seq).unsqueeze(2))
                else: 
                    out = model.forward(torch.tensor(seq).cuda().unsqueeze(2))

            output_gene[which_seq, :, count] = (
                out).cpu().detach().numpy().copy()
            count += 1

    return output_gene, debugging_starts


def process_file_locs_no_mutations(gene_list_with_pos: pd.DataFrame,
                                   destination_folder: str, model, use_cuda: bool) -> None:
    """Run DeepSea for the specified locations. Save outputs as h5py files"""
    if use_cuda:
        model = model.cuda()
    print('Start processing files')

    file_already_exists = 0
    for i, gene in gene_list_with_pos.iterrows():
        file_name = gene.gene_name+'_'+str(int(gene.tss))+'.hd5f'
        if os.path.isfile(destination_folder + file_name):
            file_already_exists += 1
            continue
        ds_output, _ = get_full_model_prediction(model=model, chrom=gene.chrom, strand=gene.strand,
                                                 center=gene.tss, mutation_pos=-1, run_ref=True, run_alt=False,
                                                 two_way_avg=True, use_cuda=use_cuda)

        f = h5py.File(destination_folder + file_name, "w")
        f.create_dataset("full_tss", data=ds_output, compression='gzip')
        f.attrs['gene_name'] = gene.gene_name
        f.attrs['tss'] = gene.tss
        f.attrs['chrom'] = gene.chrom
        f.attrs['strand'] = gene.strand
        f.attrs['cage_used'] = gene.cage_used
        f.close()

        if i % 100 == 0:
            print('Processed: ', i)

    print('Processed: ', i+1)
    print('Already existed: ', file_already_exists)


def process_variants_no_ref(variants_file: pd.DataFrame, destination_folder: str, model, use_cuda: bool) -> None:
    """ Run DeepSea for the variants, checking if the variant exists already, but not assuming any to be reference.
        Used for ldscore regression in the study.
        Save outputs into associated h5py files (for each gene).
        Assume that file is ordered by gene_name for convenience (less load/save commands)"""
    if use_cuda:
        model = model.cuda()
    print('Start processing file')
    
    running_gene_name = ''
    running_gene_tss = 0
    
    for i, variant in variants_file.iterrows():
        if i%100==0:
            print(i)
        if variant.gene_name != running_gene_name or variant.cl_chromStart!=running_gene_tss:
            if 'f' in locals():
                # if we have a file that was loaded for gene - close it
                f.close()
            
            # started on the the new gene
            running_gene_name = variant.gene_name
            running_gene_tss = variant.cl_chromStart
            new_file = running_gene_name + '_'+str(int(running_gene_tss))+'.hd5f'
            
            if os.path.exists(destination_folder+new_file):
                # open file
                try:
                    f = h5py.File(destination_folder+new_file, "r+")
                except:
                    # something was wrong when saving the file, will have to rewrite
                    f = h5py.File(destination_folder + new_file, "w")
            else:
                # create file and run reference for non-mutated gene
                f = h5py.File(destination_folder + new_file, "w")
            if 'full_tss' not in f.keys():
                ds_output, _ = get_full_model_prediction(model=model, chrom='chr'+str(variant.chrom), 
                                                         strand=variant.cl_strand,
                                                         center=variant.cl_chromStart, 
                                                         mutation_pos=-1, run_ref=True, run_alt=False,
                                                         two_way_avg=True, use_cuda=use_cuda)
                f.create_dataset("full_tss", compression='gzip', data=ds_output)
                f.attrs['gene_name'] = variant.gene_name
                f.attrs['tss'] = variant.cl_chromStart
                f.attrs['chrom'] = variant.chrom
                f.attrs['strand'] = variant.cl_strand
                f.attrs['cage_used'] = variant.cage_used
                
        for var_base in [variant.A1, variant.A2]:
            # for each of the variants, run A1 and A2
            dataset_name ="mut_"+str(variant.snp_chromStart)+'_'+str(var_base)
            if dataset_name in f.keys():
                # already existed, continue to next one
                continue
            # add variant info to this file
            ds_output, _ = get_full_model_prediction(model=model, chrom='chr'+str(variant.chrom), 
                                                             strand=variant.cl_strand,
                                                             center=variant.cl_chromStart, 
                                                             mutation_pos=variant.snp_chromStart, run_ref=False, 
                                                             run_alt=True, alt=var_base,
                                                             two_way_avg=True, use_cuda=use_cuda)

            if ds_output is None:
                # either could not access, or variant was reference
                continue
            else:
                f.create_dataset(dataset_name, compression='gzip', data=ds_output)
                
    if 'f' in locals():
        # reached end of file: close
        f.close()


def process_variants_with_ref(variants_file: pd.DataFrame, destination_folder: str, model, use_cuda: bool) -> None:
    """ Run DeepSea for the variants, checking if the variant exists already and if reference is correct.
        Save outputs into associated h5py files (for each gene).
        Assume that file is ordered by gene_name for convenience (less load/save commands)"""
    if use_cuda:
        model = model.cuda()
    print('Start processing file')
    
    running_gene_name = ''
    running_gene_tss = 0
    long_muts = 0
    var_existed = 0
    calculated_vars = 0
    
    for i, variant in variants_file.iterrows():

        if i%100==0:
            print(i)
            print('Existed variants:', var_existed)
            print('Calculated variants:', calculated_vars)

        if len(variant.ref)!=1 or len(variant.alt)!=1:
            long_muts += 1
            continue
        
        if variant.gene_name != running_gene_name or variant.tss!=running_gene_tss:
            if 'f' in locals():
                # if we have a file that was loaded for gene - close it
                f.close()
            
            # started on the the new gene
            running_gene_name = variant.gene_name
            running_gene_tss = variant.tss
            new_file = running_gene_name + '_'+str(int(running_gene_tss))+'.hd5f'
            
            if os.path.exists(destination_folder+new_file):
                # open file
                try:
                    f = h5py.File(destination_folder+new_file, "r+")
                except:
                    # something was wrong when saving the file, will have to rewrite
                    f = h5py.File(destination_folder + new_file, "w")
                try:
                    # check there are no "bad symbol table node signature"
                    check_keys = list(f.keys())
                except:
                    # something was wrong when saving the file, will have to rewrite
                    f.close()
                    f = h5py.File(destination_folder + new_file, "w")
            else:
                # create file and run reference for non-mutated gene
                f = h5py.File(destination_folder + new_file, "w")
            if 'full_tss' not in f.keys():
                ds_output, _ = get_full_model_prediction(model=model, chrom='chr'+str(variant.chrom), 
                                                         strand=variant.strand,
                                                         center=variant.tss, 
                                                         mutation_pos=-1, run_ref=True, run_alt=False,
                                                         two_way_avg=True, use_cuda=use_cuda)
                f.create_dataset("full_tss", compression='gzip', data=ds_output)
                f.attrs['gene_name'] = variant.gene_name
                f.attrs['tss'] = variant.tss
                f.attrs['chrom'] = variant.chrom
                f.attrs['strand'] = variant.strand
                f.attrs['cage_used'] = variant.cage_used
                 
        dataset_name ="mut_"+str(variant.snp_chromStart)+'_'+str(variant.alt)
        if dataset_name in f.keys():
            # already existed, continue to next one
            var_existed += 1
            continue
        # add variant info to this file
        ds_output, _ = get_full_model_prediction(model=model, chrom='chr'+str(variant.chrom), 
                                                            strand=variant.strand,
                                                            center=variant.tss, 
                                                            mutation_pos=variant.snp_chromStart, run_ref=False, 
                                                            run_alt=True, ref=variant.ref, alt=variant.alt,
                                                            two_way_avg=True, use_cuda=use_cuda)

        if ds_output is None:
            # maybe could not access
            continue
        else:
            calculated_vars += 1
            f.create_dataset(dataset_name, compression='gzip', data=ds_output)
                
    if 'f' in locals():
        # reached end of file: close
        f.close()