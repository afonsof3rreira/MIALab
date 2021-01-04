from bin.multiple_boxplots_combos import main

''' This file was used to call the scripts of the box plots used in the presentation and article:

   (1) multiple_boxplots_RF_grid_search.py was used for the grid search using only the baseline features

   (2) multiple_boxplots_fof_glcm.py was used for testing:
           • individual FOFs + baseline features
           • GLCMFs + baseline features
           • baseline features alone

   (3) multiple_boxplots_combos.py was used for the combination tests:
           • best selected FOFs + baseline features
           • best selected GLCMFs + baseline features
           • best selected FOFs + best selected GLCMFs + baseline features
           • all FOFs + all GLCMFs + baseline features
   
   Notes:
   FOFs = First Order Features
   GLCMFs = Grey Level Co-occurrence Matrix Features
   multiple_boxplots_compare_2 was only used to compare 2 methods during our initial research
'''

# input directory, output directory
test_folder_path = 'C:/Users/afons/OneDrive - Universidade de Lisboa/Erasmus/studies/MIAlab/project/Results_midterm/combo_experiments_w_baseline'
output_path = 'C:/Users/afons/OneDrive - Universidade de Lisboa/Erasmus/studies/MIAlab/project/Results_midterm/MIALab_tests/a_boxplots_presentation/combo_fofs_sofs_w_baseline_article'

main(test_folder_path, output_path)
