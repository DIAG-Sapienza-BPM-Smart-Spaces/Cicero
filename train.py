 
# ===========================================
# ||                                       ||
# ||Section 1: Importing modules           ||
# ||                                       ||
# ===========================================

import spacy
import re
import pandas as pd
import datetime
import csv
from os.path import exists
import sys

# ===========================================
# ||                                       ||
# ||Section 2: Changing csv max length     ||
# ||                                       ||
# ===========================================

csv.field_size_limit(sys.maxsize)

# ===========================================
# ||                                       ||
# ||Section 2: Checking gpu, choosing      ||
# ||             device, and model         ||
# ||                                       ||
# ===========================================

path = '/home/gueststudente/Giustizia/Pre-processing/NER_model/it_nerIta_trf/it_nerIta_trf-0.0.0'
spacy.prefer_gpu()             
nlp = spacy.load(path)
nlp2 = spacy.load("it_core_news_lg")

# ===========================================
# ||                                       ||
# ||Section 3: Cleaninig                   ||
# ||                                       ||
# ||                                       ||
# ===========================================


def cleaning_phase(path_file , name):
    """This function executes the cleaning phase. It takes as input the file to be cleaned, path file, and the name you want to give to the .csv that will be created, name."""
    judgments = reading_cleaning(path_file)
    for key in judgments: 
        judgments[key]= cleaning(judgments[key])
    crea = saving_cleaning(judgments, name)
    return None

def reading_cleaning(path_file):
    """This function takes as input a file containing multiple judgments and creates a dictionary where each key element is the identifier of a judgment and the value is the corresponding judgment in string format."""
    df = pd.read_csv(path_file, encoding = 'utf-8', sep=';')
    df = df.dropna(subset=['numerosentenza', 'annosentenza', 'parte'])
    judgments = {}
    for index, row in df.iterrows():
        key = str(int(row['numerosentenza'])) + '_' +str(int(row['annosentenza'])) + '_' +(row['parte'].replace(' ', '_'))
        value = row['text'].replace('\n', ' ')
        judgments[key] = value.split('Content-Type application/pdf')[1]
    return judgments



def cleaning(text):
    """This function takes an input string and performs the cleaning step. It removes superfluous content, changes expressions that can create problems for NLP algorithms, and performs an initial de-istantiation in a rule-based manner."""
    months = {' gennaio ': '/01/', ' febbraio ': '/02/',  ' marzo '    : '/03/', ' aprile ' : '/04/', ' maggio '  : '/05/',
        ' giugno ': '/06/', ' luglio ' : '/07/', ' agosto '  : '/08/',  ' settembre ': '/09/',
        ' ottobre ': '/10/', ' novembre ': '/11/', ' dicembre ': '/12/'}
    for i, j in months.items():
        text = re.sub(i, j, text, flags=re.IGNORECASE) 
    abbreviations = {'sig\.\s*ra': 'signora', 'sig\.\s*ri': 'signori','sig\.\s+': 'signor',
                      'avv\.\s*ti':'avvocati' , 'avv\.\s+':'avvocato',
                      'dott\.\s*ssa':'dottoressa', 'dott\.\s*ri':'dottori' , 'dott\.\s*sse':'dottoresse' , 'dott\.':'dottore',
                      'geom\.': 'geometra',
                      'est\.':'esterno',
                      'on\.': 'onorevole',
                      'cc\.dd\.': 'cosiddetto', 'c\.d\.': 'cosiddetto', 'c\.dd\.': 'cosiddetti',
                      'est\.': 'esterno',  'int\.': 'interno',
                      'coop\.': 'cooperativa',
                      'p\.es\.': 'per esempio',
                      'ecc\.': 'eccetera',
                      'lett\.': 'lettera',
                      '\s+co\.': ' comma',
                      'doc\.': 'documento',
                      'prof\.\s*ssa':'professoressa','prof\.': 'professore',
                      'cfr\.':'confronta',
                      'conf\.':'conforme',
                      'cit\.':'citazione',
                      'cap\.':'chapter',
                      'circ\.':'circolare',
                      'cost\.':'costituzione',  
                      'fasc\.':'fascicolo',  
                      'rag\.':'ragioniere', 
                      'all\.':'allegato', 
                      'p\.t\.':'pro tempore',
                       '€\.':'€ ',
                       'cass\.':'cassazione', 
                       'ss\.uu\.':'sezioni unite',
                       'c\.c\.':'codice civile',
                       'd\.lgs\.':'decreto legislativo',
                      'rel\.': 'relazione',
                        'dir\.': 'diritto',
                      'reg\.': 'regolamento',
                      'sent\.': 'judgment',
                      'delib\.': 'deliberazione',
                      'c\.t\.u\.': "consulente tecnico d'ufficio",
                      'c\.p\.c\.': 'codice procedure civile',
                      'o\.d\.g\.': 'ordine dei giornalisti',
                      'c\.p\.p\.': 'codice procedure penale',
                      'a\.s\.': 'anno scolastico',
                      'd\.l\.': 'decreto legge',
                      't\.u\.': 'text unico',
                      'vd\.|v\.': 'vedi',
                      'reg\.':'regolamento'
                        }
    for i,j in abbreviations.items():
        text = re.sub(i, j, text, flags=re.IGNORECASE)
    text = re.sub(r"\bpagina\s*\d+\s*(di\s*\d+)?|\bpage\s*(\d+)?\s*(di\s*\d+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r'(=){2,}|(-){2,}|§|\*|°|acroform|AUTVEND||•|','', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+[0-9]+/(n\.\s*)?[0-9]+[\s,.;\)]|n\.\s*[0-9]+/(n\.\s*)?[0-9]+[\s,.;\)]|numero\s*[0-9]+\s*del\s*[0-9]+', " <|NUMANN|> ", text) 
    text = re.sub(r'(C\.F\.\s*)?\s*[A-Z]{3}\s*[A-Z]{3}\s*\d{2}\s*[A-Z]\d{2}\s*[A-Z]\d{3}\s*[A-Z]', " <|CODF|> ", text, flags=re.IGNORECASE)
    text = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', " <|URL|> ", text, flags=re.IGNORECASE)
    text = re.sub(r'F\s*i\s*r\s*m\s*a\s*t\s*o\s*D\s*a\s*:\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*', " ", text)
    text = re.sub(r'(E)?\s*m es so D a\s*:\s*([A-Z0-9.]{0,2}\s+)*', ' ', text)
    text = re.sub(r'[D\s+]a:\s*([A-Z0-9.]{0,2}\s+)*', ' ', text)
    text = re.sub(r'\S\s*e\s*r\s*i\s*a\s*l\s*#\s*:\s*([a-z0-9]{0,3}\s+)*', ' ', text)
    text = re.sub(r'\bsignaturedata date \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{4}\b', '', text ,flags=re.IGNORECASE)
    text = re.sub(r"file:///[A-Z]{1}:([^ ]){1,}\s+|extlnk://([^ ]){1,}\s+", " ", text)
    text = re.sub(r"\bannotation\b", "", text, flags=re.IGNORECASE)
    return text

def saving_cleaning(judgments, name):
    """This function takes as input a dictionary containing strings, judgments and creates a .csv file. The name parameter specifies what the name of the created file will be."""
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Cleaning/'+ name +'_pulizia.csv', "w",  encoding = 'utf-8') as output:
        writer = csv.writer(output)
        for key, value in judgments.items():
            writer.writerow([key, value])
    return None


# ===========================================
# ||                                       ||
# ||Section 4: Division                    ||
# ||                                       ||
# ||                                       ||
# ===========================================

def division_phase(file):   
    """This function takes as input the output of the cleaning phase and creates two files. The first file contains the judgments that were split correctly, the second file containing the judgments that were not split correctly."""
    judgments = reading_division(file)
    check_exist(file)
    key_chapters = reading_key_chapters(file)
    incorrect_divisions = reading_incorrect_divisions(file)
    key_chapters, correct_divisions , incorrect_divisions = division(judgments, key_chapters, incorrect_divisions)
    checking = check(correct_divisions , incorrect_divisions)
    saving_division(key_chapters, incorrect_divisions, file)
    return key_chapters , checking

def reading_division(file):
    """This function takes as input the name of a file and returns a dictionary containing the, for each element, strings divided into chapters."""
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Cleaning/'+ file +'_pulizia.csv', mode='r', encoding = 'utf-8' ) as file:
        reader = csv.reader(file , delimiter=',')
        judgments = {} 
        for row in reader:
            if len(row) < 2:
                continue
            judgments[row[0]] = row[1]
    return judgments

def key(head):
    """This function creates a unique identifier given a judgment header."""
    head = head.replace(':','SEPARATORE').replace(',','SEPARATORE').split('SEPARATORE')
    head = ('_'+head[0]+'_'+head[11]+'_'+head[4]).replace(' ', '')
    return head


def divisore(judgment, dividers):
    """This function takes as input a judgment and a list of dividers. For each element in dividers, if it is present in the judgment, it replaces it with a token 'DIVIDER__' which will be used to segment the judgment."""
    for div in dividers:
        judgment = judgment.replace(div, 'DIVIDERS__' )
    last_occurrence_index = judgment.rfind('DIVIDERS__')
    if last_occurrence_index != -1:
        judgment = judgment[:last_occurrence_index] + 'DIVISORE_SEZIONE' + judgment[last_occurrence_index + len('DIVIDERS__'):] 
    judgment = judgment.replace('DIVIDERS__', ' ')
    return judgment

def reading_key_chapters(file):
    """This function reads the file containing that judgments that have already been divided into different chapters."""
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/'+ file + '_divisione.csv', 'r',  encoding = 'utf-8') as file:
        reader = csv.reader(file)
        key_chapters = list(reader)
    return key_chapters

def check_exist(file):
    """This function checks if a split file already exists. If it does not exist it creates it."""
    if not exists('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/'+ file + '_divisione.csv'):
        create = open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/'+ file + '_divisione.csv', 'x',  encoding = 'utf-8')
    if not exists("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/" + file + "_divisioni_sbagliate.txt"):
        create = open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/" + file + "_divisioni_sbagliate.txt", 'x',  encoding = 'utf-8')
    return None

def reading_incorrect_divisions(file):
    """This function is used to read the file containing the judgments that were not split correctly."""
    with open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/" + file + "_divisioni_sbagliate.txt", "r", encoding = 'utf-8') as file:
        incorrect_divisions = file.read().splitlines()
    return incorrect_divisions

def division(judgments, key_chapters, incorrect_divisions):
    """This function takes as input judgments that need to be divided into chapters, judgments that have been correctly divided, and judgments that have been incorrectly divided, and adds the new judgments to either the first group or the second group."""
    judgments_csv = [col[0] for col in key_chapters]
    for key in judgments:
        if key in judgments_csv:
            continue
        else:
            judgment , errore  =  divisione_capitoli(judgments[key])
            if errore:
                incorrect_divisions.append(key)
            else:
                key_chapters.append([key, judgment[0], judgment[1], judgment[2]])
                if key in incorrect_divisions:
                    incorrect_divisions.remove(key)
    judgments_csv = [col[0] for col in key_chapters]
    return key_chapters, judgments_csv, incorrect_divisions

def check(correct_divisions , incorrect_divisions):
    """This function takes as input the list of correctly split sentences and incorrectly split sentences. It checks for identifiers of judgments present in both sets."""
    intersection = list(set(correct_divisions) & set(incorrect_divisions))
    if intersection != []:
        print ('errore check')
        return intersection
    return None

def saving_division(key_chapters , incorrect_divisions , name):
    """Questa funzione prende in input le sentenze correttamente divise, le sentenze non correttamente e salva in formato .csv e .txt. Il terzo parametro indica quale nome attribuire ai due file."""
    incorrect_divisions = [key.replace('\n', '') for key in incorrect_divisions]
    incorrect_divisions = [key for key in incorrect_divisions if key != '']
    incorrect_divisions = set(incorrect_divisions)
    with open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/" + name + "_divisioni_sbagliate.txt", "w", encoding = 'utf-8') as output:
        for item in incorrect_divisions:
            output.write("%s\n" % item)
    
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/'+ name + '_divisione.csv' , 'w', newline='',  encoding = 'utf-8') as name:
        writer = csv.writer(name)
        writer.writerows(key_chapters)

    return None




def divisione_capitoli(judgment):
    """This function takes as input correctly split judgments, incorrectly split judgments, and saves in .csv and .txt format. The third parameter indicates what name to give to the two files."""
    divider1 = ['MOTIVI DELLA DECISIONE',
                  'RAGIONI IN FATTO E DIRITTO DELLA DECISIONE',
                  'FATTO E DIRITTO',
                  'RAGIONI IN FATTO ED IN DIRITTO DELLA DECISIONE',
                  'MOTIVI IN FATTO E IN DIRITTO',
                  'Svolgimento del processo e motivi della decisione',
                  'S V O L G I M E N T O D E L P R O C E S S O', 
                  'RAGIONI DELLA DECISIONE',
                  'Motivi della decisione' ,
                  'Svolgimento del processo',
                  'C O N S I D E R A T O',
                  'MOTIVAZIONE',
                  'CONCLUSIONI DELLE PARTI',
                  'CON LA PARTECIPAZIONE DEL PUBBLICO MINISTERO ',
                  'SVOLGIMENTO DEL PROC ESSO', 
                  'svolgimento del processo',
                  'RAGIONI DI FATTO E DI DIRITTO  DELLA  DECISIONE',
                  'RAGIONI DI FATTO E DI DIRITTO',
                 'MOTIVI DELLA DECISIONE IN FATTO E DIRITTO',
                  'CONCLUSIONI',
                 'Conclusioni rassegnate congiuntamente dalle parti:',
                 'N O N C H E']
    divider2 = [  'P.Q.M',
                  'P. Q. M',
                  'P. Q. M',
                  'P.   Q.   M',
                  'M O T I V I D E L L A D E C I S I O N E',
                  'P Q M',
                 'p.q.m' ,
                  'PQM' , 
                  'P.Q.M',
                  'P.Q.M.',
                  'P.Q.M.']  
    judgment = divisore(judgment, divider1)
    judgment = divisore(judgment, divider2)  
    judgment = judgment.split('DIVISORE_SEZIONE')
    if len(judgment) != 3:
        return judgment , True
    else:
        return judgment , False


# ===========================================
# ||                                       ||
# ||Section 5: Anonymization               ||
# ||                                       ||
# ||                                       ||
# ===========================================


def deinstantiation_phase(file, chapter):
    """This function takes as input the output of the division phase and the chapter you want to de-instantiate. The output is the text of the de-istanced file. In addition, the function creates a .txt file that saves the de-instantiated text."""
    chunks = reading_deinstantiation(file, chapter)
    text = deinstantiation(chunks)
    text = token(text)
    saving_deinstantiation(text, file, chapter) 
    return text

def reading_deinstantiation(file, chapter):
    """This function takes as input the output of the division phase and the chapter you want to de-instantiate. The function returns, for each sentence in the file, only the desired chapters."""
    read = reading_key_chapters(file)
    chunks = []
    for judgment in read:
        judgment = judgment[int(chapter)]
        chunks.append(judgment)
    return chunks    

def deinstantiation(chapters):
    """This function takes as input the output of 'reading_deinstantiation(file, chapter)' and de-instantiates the data using the spacy pattern named 'nlp'. The output is a string."""
    text = ''
    for text in chapters:

        doc = nlp(text) 
        for e in reversed(doc.ents):
            start = e.start_char
            end = start + len(e.text)
            if e.label_ in ["WORK_OF_ART" , "LANGUAGE"]:
                continue
            text = text[:start] + e.label_ + text[end:]
        text +=  text
        text += '\n' 
    return text

def token(text):
    """This function takes as input the ouput of the 'deinstantiation(chapters)' function and modifies the labels to facilitate the training phase."""
    labels = ["PERCENT", "PER", "NORP", "ORG", "GPE", "LOC", "DATE", "MONEY","FAC", "PRODUCT", "EVENT",
              "LAW", "TIME", "QUANTITY", "ORDINAL", "CARDINAL", "LANGUAGE"]
    for label in labels:
        text = re.sub(r'(\b'+label+ '\s*){1,}', ' <|'+ label + '|> ' , text)
    return(text)

def saving_deinstantiation(text,file, chapter):
    """This function takes as input the uotput of the 'token(text)' function, the name you want to give to the file to be created, and the capitil that has been de-istanced. The function creates a .txt file containing the de-istanced sentences."""
    with open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/" + file + "_"  + chapter + ".txt", "w",  encoding = 'utf-8') as output:  
        output.write("%s\n" % text)
    output.close()
    return None

# ===========================================
# ||                                       ||
# ||Section 6: Creating Dataframe          ||
# ||                                       ||
# ||                                       ||
# ===========================================

def create_dataframe(text_files, chapter,  length = '600'):
    """This function takes as input a file containing the de-spaced sentences, the chapters that are in the file, and a length. The function will create a file that splits the chapters into chunks of the maximum length 'length'."""
    phrases = []
    for i, text_file in enumerate(text_files):
        with open(text_file, encoding = 'utf-8') as file:
            content = file.read()
            phrases.extend(content.split('\n'))
    phrases = min_length(phrases)
    df = pd.DataFrame({'Capitoli': phrases}, index=range(len(phrases)))
    df['Capitoli'] = df['Capitoli'].str.replace('\n', '')
    outliers = []
    for i in range(len(df)):
        if len(df.loc[i]['Capitoli']) > int(length):
            outliers.append(i) 
    df = df.drop(outliers)
    df.reset_index()
    df.to_csv('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Merging/merge_' + chapter + '_'+ length + '.csv', index=False, encoding = 'utf-8' )
    return df

def min_length(phrases):
    """This function takes as input a list of strings. Each string is a chapter of a sentence. The function divides each chapter into sentences. The function saves only sentences containing more than 10 words."""
    final_document = []
    for chapter in phrases:
      doc = nlp2(chapter)
      for sent in doc.sents:
        if len(sent.text.split()) > 10:
            final_document.append(sent.text)
        else:
            continue
    return final_document


# ===========================================
# ||                                       ||
# ||Section 7: Executing preprocessing     ||
# ||           pipeline                    ||
# ||                                       ||
# ===========================================

chapter = '2'


for i in [100]: 
    i = str(i)

    if not exists('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/'+ i + '.csv'):
        continue
    print(i)
    clean = cleaning_phase('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/'+ i + '.csv',i)
    print(i, 'ok1')
    div = division_phase(i)
    print(i, 'ok2')
    anon = deinstantiation_phase(i, chapter)
    print(i, 'ok3')
    files = []

for i in [100]:
    if not exists('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/'+str(i)+'_'+ chapter +'.txt'):
        continue

    files.append('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/'+str(i)+'_'+ chapter +'.txt')
