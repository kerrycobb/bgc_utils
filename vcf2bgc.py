#!/usr/bin/env python

## Create BGC input files from VCF and list of samples for each parent
## and the admixed population

import fire
import allel
import pandas as pd

def get_sample_ids(path):
    ## Read sample ids from file and return as list 
     lines = []
     with open(path) as fh:
         for i in fh.readlines():
             lines.append(i.strip())
     return lines

def run(vcf_path, samples_path, out_path, parent_pop=True):
    print("Outputting bgc file to {}.bgc".format(out_path))
    samples = get_sample_ids(samples_path)
    vcf = allel.read_vcf(vcf_path, samples=samples, fills=dict(GT=-9))
    arr = allel.GenotypeArray(vcf["calldata/GT"])
    if parent_pop:
        alleles = arr.count_alleles()
    else:
        sample_order = pd.DataFrame(vcf["samples"])
        sample_order.to_csv("{}.sample-id-map.csv".format(out_path), header=None)
    with open(out_path, "w") as fh:
        for i in range(arr.shape[0]):
            fh.write("locus_{}\n".format(i))
            if parent_pop:
                fh.write(" ".join(alleles[i].astype(str)))
                fh.write("\n")
            else:
                fh.write("pop_0\n")
                sum = arr[i].sum(axis=1)
                for j in range(arr.shape[1]):
                    if sum[j] == 0:
                        fh.write("2 0\n") 
                    elif sum[j] == 1:
                        fh.write("1 1\n") 
                    elif sum[j] == 2:
                        fh.write("0 2\n") 
                    elif sum[j] == -18:
                        fh.write("-9 -9\n")
                    else:
                        quit("Error. Invalid genotypes.")

if __name__ == "__main__":
    fire.Fire(run)