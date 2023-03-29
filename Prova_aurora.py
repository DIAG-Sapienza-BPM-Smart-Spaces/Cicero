#!/usr/bin/env python
# coding: utf-8



import spacy
import re
import pandas as pd
import datetime
import csv




path_Mattia = '/home/gueststudente/Giustizia/Pre-processing/NER_model/it_nerIta_trf/it_nerIta_trf-0.0.0'
spacy.prefer_gpu()             
nlp = spacy.load(path_Mattia)




def pulizia(path_file , nome):
    sentenze = lettura_pulizia(path_file)
    for key in sentenze: 
        sentenze[key]= elimina(sentenze[key])
    crea = trascrizione_pulizia(sentenze , nome)
    return None

def lettura_pulizia(path_file):
    df = pd.read_csv(path_file, encoding = 'utf-8', sep=';')
    df = df.dropna(subset=['numerosentenza', 'annosentenza', 'parte'])
    sentenze = {}
    #conta = 1
    for index, row in df.iterrows():
        #if conta == 10:
         #   break
        #conta +=1
        key = str(int(row['numerosentenza'])) + '_' +str(int(row['annosentenza'])) + '_' +(row['parte'].replace(' ', '_'))
        value = row['text'].replace('\n', ' ')
        sentenze[key] = value.split('Content-Type application/pdf')[1]
    return sentenze



def elimina(testo):
    mesi = {' gennaio ': '/01/', ' febbraio ': '/02/',  ' marzo '    : '/03/', ' aprile ' : '/04/', ' maggio '  : '/05/',
        ' giugno ': '/06/', ' luglio ' : '/07/', ' agosto '  : '/08/',  ' settembre ': '/09/',
        ' ottobre ': '/10/', ' novembre ': '/11/', ' dicembre ': '/12/'}
    for i, j in mesi.items():
        testo = re.sub(i, j, testo, flags=re.IGNORECASE) 
    abbreviazioni = {'sig. ': 'sig.', 'avv. ':'avv.' , 'dott. ':'dott.', 'geom. ': 'geom.', 'est. ':'est.' , 'On. ': 'On.'}
    for i,j in abbreviazioni.items():
        testo = re.sub(i, j, testo, flags=re.IGNORECASE)
    testo = re.sub(r"\bpagina\s*\d+\s*(di\s*\d+)?|\bpage\s*(\d+)?\s*(di\s*\d+)?", "", testo, flags=re.IGNORECASE)
    testo = re.sub(r'(=){2,}|(-){2,}|§|\*|°|acroform|AUTVEND||•','', testo, flags=re.IGNORECASE)
    testo = re.sub(r'\s+', ' ', testo)
    testo = re.sub(r'\s+[0-9]+/(n\.\s*)?[0-9]+[\s,.;\)]|n\.\s*[0-9]+/(n\.\s*)?[0-9]+[\s,.;\)]|numero\s*[0-9]+\s*del\s*[0-9]+', " Numero/Anno ", testo) #NUMERO/ANNO
    testo = re.sub(r'(C\.F\.\s*)?\s*[A-Z]{3}\s*[A-Z]{3}\s*\d{2}\s*[A-Z]\d{2}\s*[A-Z]\d{3}\s*[A-Z]', " Cod_F ", testo, flags=re.IGNORECASE) #CODICE FISCALE
    testo = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', " URL ", testo, flags=re.IGNORECASE) #URL
    testo = re.sub(r'F\s*i\s*r\s*m\s*a\s*t\s*o\s*D\s*a\s*:\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*[A-Z]*\s*', " ", testo)
    testo = re.sub(r'(E)?\s*m es so D a\s*:\s*([A-Z0-9.]{0,2}\s+)*', ' ', testo)
    testo = re.sub(r'[D\s+]a:\s*([A-Z0-9.]{0,2}\s+)*', ' ', testo)
    testo = re.sub(r'\S\s*e\s*r\s*i\s*a\s*l\s*#\s*:\s*([a-z0-9]{0,3}\s+)*', ' ', testo)
    testo = re.sub(r'\bsignaturedata date \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{4}\b', '', testo ,flags=re.IGNORECASE)
    return testo

def trascrizione_pulizia(sentenze, nome):
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Cleaning/'+ nome +'_pulizia.csv', "w",  encoding = 'utf-8') as output:
        writer = csv.writer(output)
        for key, value in sentenze.items():
            writer.writerow([key, value])
    return None





def divisione(file):    
    sentenze = lettura_divisione(file)
    chiave_e_capitoli = lettura_chiave_e_capitoli(file)
    sentenze_sbagliate = lettura_sentenze_sbagliate(file)
    chiave_e_capitoli, sentenze_corrette , sentenze_sbagliate = capitoli(sentenze, chiave_e_capitoli, sentenze_sbagliate)
    controllo = check(sentenze_corrette , sentenze_sbagliate)
    trascrizione_divisione(chiave_e_capitoli, sentenze_sbagliate, file)
    return chiave_e_capitoli , controllo

def lettura_divisione(file):
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Cleaning/'+ file +'_pulizia.csv', mode='r', encoding = 'utf-8' ) as file:
        reader = csv.reader(file , delimiter=',')
        sentenze = {} 
        for row in reader:
            if len(row) < 2:
                continue
            sentenze[row[0]] = row[1]
    return sentenze



def chiave(intestazione):
    # manca ancora l'estrattore della città che serve come chiave
    intestazione = intestazione.replace(':','SEPARATORE').replace(',','SEPARATORE').split('SEPARATORE')
    intestazione = ('_'+intestazione[0]+'_'+intestazione[11]+'_'+intestazione[4]).replace(' ', '')
    return intestazione


def divisore(sentenza, divisori):
    for div in divisori:
        sentenza = sentenza.replace(div, 'DIVISORI__' )
    last_occurrence_index = sentenza.rfind('DIVISORI__')
    if last_occurrence_index != -1:
        sentenza = sentenza[:last_occurrence_index] + 'DIVISORE_SEZIONE' + sentenza[last_occurrence_index + len('DIVISORI__'):] 
    sentenza = sentenza.replace('DIVISORI__', ' ')
    return sentenza

def lettura_chiave_e_capitoli(file):
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/'+ file + '_divisione.csv', 'r',  encoding = 'utf-8') as file:
        reader = csv.reader(file)
        chiave_e_capitoli = list(reader)
    return chiave_e_capitoli

def lettura_sentenze_sbagliate(file):
    with open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/" + file + "_divisioni_sbagliate.txt", "r", encoding = 'utf-8') as file:
        sentenze_sbagliate = file.read().splitlines()
    return sentenze_sbagliate

def capitoli(sentenze, chiave_e_capitoli, sentenze_sbagliate):
    sentenze_csv = [col[0] for col in chiave_e_capitoli]
    for key in sentenze:
        if key in sentenze_csv:
            continue
        else:
            sentenza , errore  =  divisione_capitoli(sentenze[key])
            #print(len(sentenza))
            if errore:
                sentenze_sbagliate.append(key)
            else:
                #print(key) mi serve per vedere cosa recupera in caso quando lavoro sulle divisioni sbagliate
                chiave_e_capitoli.append([key, sentenza[0], sentenza[1], sentenza[2]])
                if key in sentenze_sbagliate:
                    sentenze_sbagliate.remove(key)
    sentenze_csv = [col[0] for col in chiave_e_capitoli]
    return chiave_e_capitoli, sentenze_csv, sentenze_sbagliate

def check(sentenze_corrette , sentenze_sbagliate):
    intersection = list(set(sentenze_corrette) & set(sentenze_sbagliate))
    if intersection != []:
        print ('errore check')
        return intersection
    return None

def trascrizione_divisione(chiave_e_capitoli , sentenze_sbagliate , file):
    sentenze_sbagliate = [elemento.replace('\n', '') for elemento in sentenze_sbagliate]
    sentenze_sbagliate = [elemento for elemento in sentenze_sbagliate if elemento != '']
    sentenze_sbagliate = set(sentenze_sbagliate)
    with open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/" + file + "_divisioni_sbagliate.txt", "w", encoding = 'utf-8') as output:
        for item in sentenze_sbagliate:
            output.write("%s\n" % item)
    
    with open('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Division/'+ file + '_divisione.csv' , 'w', newline='',  encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(chiave_e_capitoli)

    return None




def divisione_capitoli(sentenza):
    #sentenza = re.sub("\s*MOTIVIDELLADECISIONE\s*", "MOTIVI DELLA DECISIONE", sentenza, flags=re.IGNORECASE)
    #sentenza = re.sub("\s*RAGIONIINFATTOEDIRITTODELLADECISIONE\s*", "RAGIONI IN FATTO E DIRITTO DELLA DECISIONE", sentenza, flags=re.IGNORECASE)
    #sentenza = re.sub("\s*FATTOEDIRITTO\s*", "FATTO E DIRITTO", sentenza, flags=re.IGNORECASE)

    divisori_1 = ['MOTIVI DELLA DECISIONE',
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
                 'N O N C H E'] #ed affini (attenzione a svolgmento e motivi)
    #provare ad aggiungere "N O N C H E'" e togliere "CONCLUSIONI"  
    # 'conclusioni delle parti :' 'CONCLUSIONI' ,'Conclusioni delle parti:'
    divisori_2 = [ 'P.Q.M',
                  'P. Q. M',
                  'P. Q. M',
                  'P.   Q.   M',
                  'M O T I V I D E L L A D E C I S I O N E',
                  'P Q M',
                 'p.q.m' ,
                  'PQM' , 
                  'P.Q.M',
                  'P.Q.M.',
                  'P.Q.M.'] #ed affini  
    sentenza = divisore(sentenza, divisori_1)
    sentenza = divisore(sentenza, divisori_2)  #QUI DEVO CONTARE QUANTE VOLTE AVVIENE IL REPLACEMENT e segnarlo in 
                                                #in qualche modo
    sentenza = sentenza.split('DIVISORE_SEZIONE')
    if len(sentenza) != 3: #per ora il numero  atteso è 3, poi si vedrà
        return sentenza , True
    else:
        return sentenza , False





def anonimizzazione(file, capitolo):
    frasi = lettura_anonimizzazione(file, capitolo)
    testo = anonimizzazione_frasi(frasi)
    testo =accorpa(testo)
    trascrizione_anonimizzazione(testo, file, capitolo) 
    return testo

def lettura_anonimizzazione(file, capitolo):
    var_lettura = lettura_chiave_e_capitoli(file)
    frasi = []
    for sentenza in var_lettura:
        sentenza = sentenza[int(capitolo)]
        #sentenza = sentenza.replace(',', ' ,').replace('.', ' .').replace(':', ' :').replace(';', ' ;').replace( '(', '( ').replace(')', ' )')
        frasi.append(sentenza)
    return frasi    

def anonimizzazione_frasi(lista):
    testo = ''
    for text in lista:
        print('indice: ', lista.index(text))
        doc = nlp(text) #
        for e in reversed(doc.ents):
            start = e.start_char
            end = start + len(e.text)
            text = text[:start] + e.label_ + text[end:]
        testo +=  text
        testo += '\n' # '\n\n'
    return testo#.replace(' .', '.').replace(' ;', ';').replace(' :', ':').replace(' ,', ',').replace( '( ', '(').replace(' )', ')')

def accorpa(testo):
    etichette = ["PER", "NORP", "ORG", "GPE", "LOC", "DATE", "MONEY", "FAC", "PRODUCT", "EVENT",
                     "LAW", "TIME", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL", "WORK_OF_ART", "LANGUAGE"]
    for etichetta in etichette:
        etichetta = etichetta
        testo = re.sub(r'('+etichetta+ '\s*){2,}', etichetta + ' ' , testo)
    return(testo)

def trascrizione_anonimizzazione(testo ,file, capitolo):
    with open("/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/" + file + "_"  + capitolo + ".txt", "w",  encoding = 'utf-8') as output:  
        output.write("%s\n" % testo)
    output.close()
    return None

def create_dataframe(text_files, capitolo,  length = '99999'):
    phrases = []
    conta = 0
    for i, text_file in enumerate(text_files):
        with open(text_file, encoding = 'utf-8') as file:
            content = file.read()
            phrases.extend(content.split('\n')) #se non va '\n\n\n\n'
    phrases = max_lunghezza(phrases, length)
    df = pd.DataFrame({'Capitoli': phrases}, index=range(len(phrases)))
    df['Capitoli'] = df['Capitoli'].str.replace('\n', '')    
    df.to_csv('/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/Merging/merge_' + capitolo + '_'+ length + '.csv', index=False, encoding = 'utf-8' )
    return df

def max_lunghezza(phrases, length = '99999'):
    lista_finale = ['']
    for capitolo in phrases:
      doc = nlp2(capitolo)
      for sent in doc.sents:
        if len(sent.text.split()) > 5:
            lista_finale.append(sent.text)
        else:
            lista_finale[-1] += ' ' + sent.text
    return lista_finale

nlp2 = spacy.load("it_core_news_lg")


start_time = datetime.datetime.now()
print('Merging ora inizio: ', start_time )
files = [#'/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/0_2.txt',
        '/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/1_2.txt',
        '/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/2_2.txt',
        '/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/3_2.txt',
        '/home/gueststudente/Giustizia/Pre-processing/Pipeline_files/De-istantiation/4_2.txt',
]
merging = create_dataframe(files, '2')
end_time = datetime.datetime.now()
elapsed_time = end_time - start_time
print("Merging tempo trascorso:", elapsed_time)


if False:
    print(' ')

    start_time = datetime.datetime.now()
    print('Pulizia ora inizio: ', start_time )
    #clean = pulizia('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/GLSA/0.csv', "0")
    clean = pulizia('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/GLSA/1.csv', "1")
    clean = pulizia('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/GLSA/2.csv', "2")
    clean = pulizia('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/GLSA/3.csv', "3")
    clean = pulizia('/home/gueststudente/Giustizia/Pre-processing/Original_sentences/GLSA/4.csv', "4")
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Pulizia Tempo trascorso:", elapsed_time)


    start_time = datetime.datetime.now()
    print('Divisione ora inizio: ', start_time )
    #division = divisione("0")
    division = divisione("1")
    division = divisione("2")
    division = divisione("3")
    division = divisione("4")
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Divisione tempo trascorso:", elapsed_time)

    start_time = datetime.datetime.now()
    print('Anonimizzazione ora inizio: ', start_time )
    #anonymizationa = anonimizzazione("0", '2')
    anonymizationa = anonimizzazione("1", '2')
    anonymizationa = anonimizzazione("2", '2')
    anonymizationa = anonimizzazione("3", '2')
    anonymizationa = anonimizzazione("4", '2')
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("Anonimizzazione tempo trascorso:", elapsed_time)



