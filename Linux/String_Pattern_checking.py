import re
import pandas as pd
from module_csv import *
#Q1) import the module and invoking the functions
# load_csv()
# calculate_rows()
# calculate_columns()
# missing_value()
#Q2)create json module and load the file and set True for who scored highest
df =pd.read_json("C:\\Users\\User\\PycharmProjects\\pythonProject2\\venv\\cricket.json")
print(df)
max_score = df["player_score"].max()
print(max_score)
df.loc[df["player_score"]==max_score,"man_0f_the_match"]=True
print(df)
#Q4)Test if a DNA sequence contains an EcoRI restriction site
import re
def res_site(dna):
    pattern =re.compile('GAATTC')
    #pattern2 = re.compile('ATTCC')
    m=pattern.search(dna)
    if m:
        print("restriction site for ecor1 is there")
    else:
        print("not there")

#Q5)presence of an AvaII recognition site, which can have two different sequences: GGACC and GGTCC
pattern = GGACC and GGTCC
def res_site_ava(dna):
    pattern = re.compile('GGACC' or 'GGTCC')
    n = pattern.search(dna)
    if n :
        print("restriction site for AvaI1 is present in dna")
    else:
        print("no recogition site for AvaI1 in dna")
# Q6)Check for the presence of a BisI restriction site
pattern = GCNGC, where,N  =any bases A, T, G, C
def rest_site_bsi(dna):
    pattern = re.compile('GC[ATGC]')
    n= pattern.search(dna)
    if n:
        print("recongition site for Bsi is present in dna")
    else:
        print("recongition site for Bsi is not  there in dna")
#Q7) pattern to identify full-length eukaryotic messenger RNA sequences
# ^AUG[AUGC]{30,1000}A{5,10}$
#1 sequence starts with AUG
#2 after AUG 30 -1000 repeats of  any of the among bases A,U,G,C
#3 end with 5-10 A bases
#Q8)determine whether or not it contains any ambiguous bases i.e. any bases that are not A, T, G or C
def check_ambiguous(dna1):
    if re.search("[^ATGC]", dna1):
        print("ambiguous base")
    else:
        print("ambiguous base not present")
#Q9)extract the genus name and species name into separate variables
def scientific_name(sci_name):
    S = re.search("(.+) (.+)", sci_name)
    if S:
        genus = S.group(1)
        species = S.group(2)
        print("genus is:",genus)
        print("species is:",species)
#Q10) take DNA sequence and determine whether or not it contains any ambiguous bases bases are not A,T,G,C
def pos_ambigiuous(dna2):
    m =  re.search("[^ATGC]", dna2)
    if m:
        print("ambiguous base found")
        print("Ambigous base found at " + str(m.start()) + "th position")
#Q11)DNA sequence with the extract bits of string
def string_match(dna3):
    strg = re.findall("[AT]{6,}", dna3)
    print(strg)
# Q12)Write a regular expression to split the DNA string wherever we see a base that isn't A, T, G or C
# output =['ACT', 'GCAT', 'GCTACGT', 'ACGAT', 'CGA', 'TCG']
def split_dna(dna4):
    m = (re.split("[^ATGC]", dna4))
    if m :
        print(m)
#Q13)print the accession numbers
accessions = ['xkn59438', 'yhdck2', 'eihd39d9', 'chdsye847', 'hedle3455', 'xjhd53e', '45da', 'de37dp']
a = re.compile(".*5.*")
print("Accessions that contain 5:")
for id in accessions:
    if a.match(id):
        print(id)

b = re.compile(".*[de].*")
print("Accessions containing d or e :")
for id in accessions:
    if b.match(id):
        print(id)
c = re.compile(".*de.*")
print("Accession containing d and e in order:")
for id in accessions:
    if c.match(id):
        print(id)

d = re.compile(".*d[a-z]e.*")
print("Accessions contain the letters d and e in that order with a single letter between them:")
for id in accessions:
    if d.match(id):
        print(id)

e = re.compile("(.*d.*)(.*e.*)")
print("accession that contain both the letters d and e in any order:")
for id in accessions:
    if e.match(id):
        print(id)

f = re.compile("^[xy]")
print("Accessions start with x or y:")
for id in accessions:
    if f.match(id):
        print(id)

g = re.compile("(^[xy].*e$)")
print("Accessions start with x or y and end with e:")
for id in accessions:
    if g.match(id):
        print(id)

#h = re.compile(".*[0-9][0-9][0-9].*")
h = re.compile(".*\d{3}.*")
print("Accession contain three or more digits in a row:")
for id in accessions:
    if h.match(id):
        print(id)

i = re.compile(".*d[arp]$")
print("Accessions end with d followed by either a, r or p:")
for id in accessions:
    if i.match(id):
        print(id)


#14]Predict the fragment lengths that we will get if we digest the sequence with two made up restriction enzymes –
# AbcI, whose recognition site is ANT/AAT, and AbcII, whose recognition site is GCRW/TG.
abc1 - ANT/AAT,where N=[ATGC]
abc2 - GCRW/TG. where R = [GA] and W=[AT]
dna = open("dna.txt").read() #read the dna.txt file
#print(str(len(dna)))  #length of the dna
cutsite = [0]
for match in re.finditer(r"A[ATGC]TAAT", dna): #iterating over recognition site
    cutsite.append(match.start() + 3)  #cuts at ANT ,from the beginning  3 bits and appending sequence length to cutsite
#print(cutsite)
for match in re.finditer(r"GC[AG][AT]TG", dna): #iterating over recognition site
    cutsite.append(match.start() + 4) #cuts at GCRW ,from the beginning 4 bits and appending sequence length to cutsite
#print(cutsite)
cutsite.append(len(dna)) #appending length of dna to cutsite
sorted_cuts = sorted(cutsite) #sorting
#print(sorted_cuts)
for i in range(1, len(sorted_cuts)):
    fragment_size = sorted_cuts[i] - sorted_cuts[i - 1]
    print("one fragment size is " + str(fragment_size))
def main():
    # dna = "ATCGCGAATTCAC"
    # dna = "CTAAGGACCTTCAGGTCC"
    DNA = "AACTGGTCCTCCCC"
    dna = "ATCGCGAATTCAC"
    dna1 = "ATCGCGYAATTCAC"
   # sci_name ="Drosophila melanogaster"
    sci_name ="Homo sapiens"
    dna2 = "CGATNCGGAACGATC"
    dna3="ACTGCATTATATCGTACGAAATTATACGCGCG"
    dna4 = "ACTNGCATRGCTACGTYACGATSCGAWTCG"
   # scientific_name(sci_name)
    #res_site(dna)
    #res_site_ava(dna)
    # rest_site_bsi(dna)
    #check_ambiguous(dna1)
    #pos_ambigiuous(dna2)
    #string_match(dna3)
   # split_dna(dna4)
if __name__=="__main__":
    main()

