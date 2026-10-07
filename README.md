# clustering_species_richness


Scripts and data used for the analysis pipeline in the 2026 MEE submission "Unsupervised source separation and clustering of soundscape recordings predicts avian richness in hyperdiverse systems".
For now, the slurm (.sh) files to execute these python scripts in a HPC are not included, as these settings will vary depending on your local cluster, but they are available upon request.


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

11: perch_v1_v2_calibration.py

12: index_calculator.py

13: perch_sr_counter.py

14: flowchart_image_gen.py

15: figs.rmd


The data (SR, cluster counts, index values, Perch SR, etc, for each recording/site/etc) is saved as "data.csv". 
