# clustering_species_richness


Scripts used for the analysis pipeline in [name of paper about source separation and clustering estimating species richness]

for now, i don't include the slurm (.sh) files to excute these python scripts in the HPC - doesn't seem necessary. But available upon request


The order these scripts are run:

1: recording_site_sr_generator.py

2: split_files.sh

3: band_pass.py

4: source_separation.py

5: remove_noise.py

6: count_active_channels.py

7: generate embeddings (using bacpipe- not included. see github.com/bioacoustic-ai/bacpipe)

8: cluster_embeddings.py

9: cluster_populator.py

10: perch_classification.py

11: perch_sr_counter.py

12: index_calculator.py

13: flowchart_image_gen.py

14: cluster_paper_plots.rmd


