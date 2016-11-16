a.) Split the 20gb transaction file into multiple chunks(approximately 4gb each)
b.) On every individual chunk run the DataClean script, which results in cleaned output for the individual chunk;
c.) Merge all the outputs into a single file
d.) Proceed to perform reductions on the merged output file.
