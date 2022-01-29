
Add your text file here to train the models. 

For example, you could use the Project Gutenberg eBook "The Complete Works of William Shakespeare" by, none other than, William Shakespeare himself. The file can be found here: <https://www.gutenberg.org/files/100/100-0.txt>.

If you are going to reuse this file, you'll need to manually remove the header and the table of contents down to where "THE SONNETS" begin. Also, scroll to the end of the file and remove the "END" section of the eBook, unless you want your model to learn modern legal words in addition to Shakespeare. After these manual updates there are just over 170,000 lines in stanza form, about eight to nine words per line, and the file is 5.46 MB. The code has a clean_text() function to further clean this file and remove additional Shakespearian direction like "[_Exit Countess._]", "[_Aside._]", etc. 


In the .\code\ _Train.py files set INPUT_FILE and new OUTPUT_FILE to your text files in this subdir. Example:

INPUT_FILE = '..\\data\\Complete_Shakespeare.txt'
OUTPUT_FILE = '..\\data\\Complete_Shakespeare_cleaned.txt'


The _Predict.py files re-use the previously cleaned input file, for example:

INPUT_FILE = '..\\data\\Complete_Shakespeare_cleaned.txt'

