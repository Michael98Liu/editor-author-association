import pandas as pd
import glob

MAGDIR = '/scratch/fl1092/MAG/2021-12-06/'
PROJDIR = '/scratch/fl1092/followup-editors/'

def loadPaperEditor(init=False, dropDup=True):
    
    if init:
        papEditor = pd.concat(
            [(
                pd.read_csv(file,sep='\t')
                .assign(publisher=file.split('/')[-2].lower())
            )
             for file in glob.glob('/scratch/fl1092/*/PaperIdEditorId.csv')],
            ignore_index=True, sort=False
        )

        assert(papEditor.duplicated().any() == False)
        
        papEditor.to_csv(PROJDIR + 'PaperEditors.csv',sep='\t',index=False)
    else:
        papEditor = pd.read_csv(PROJDIR + 'PaperEditors.csv',sep='\t')

    dup = papEditor[papEditor.PaperId.duplicated(keep=False)]
    
    if dropDup:
        papEditor = papEditor.drop_duplicates(subset=['PaperId'], keep=False)
    
    return papEditor


def loadPaperAuthor(aff=False):

    columns = ['PaperId', 'AuthorId']
    if aff:
        columns.append('AffiliationId')
        
    papAu = (
        pd.read_csv(MAGDIR+"mag/PaperAuthorAffiliations.txt", sep="\t",
                    names = ['PaperId', 'AuthorId', 'AffiliationId', 'AuthorSequenceNumber',
                             'OriginalAuthor', 'OriginalAffiliation'],
                    usecols = columns,
                    dtype = {'PaperId':int, 'AuthorId':int, 'AffiliationId':float}, memory_map=True)
        .drop_duplicates()
    )
    
    return papAu

def loadPaperInfo():
    
    info = pd.read_csv(PROJDIR + 'PaperInfoGathered.csv', sep='\t',
                   usecols=['PaperId','Publisher','Journal'])
    
    return info

def loadJournals():
    
    journals = pd.concat(
        [pd.read_csv(file,sep='\t',dtype={'JournalId':float}).assign(Publisher = file.split('/')[3].lower())
         .dropna().assign(JournalId = lambda df: df.JournalId.astype(int))
         for file in glob.glob('/scratch/fl1092/*/JournalToJournalIdMapping.csv')],
        ignore_index=True, sort=False
    ).drop_duplicates(subset=['JournalId'], keep=False)
    
    return journals

def loadPaperDelay(percentage=True, paperYear=None, normalize=False):
    
    if paperYear is None:
        paperYear, _ = loadPaperRecvAcptTime()
        paperYear = paperYear.drop('RecvDate', axis=1)
    
    info = pd.read_csv(PROJDIR + 'PaperInfoGathered.csv', sep='\t',
                   usecols=['PaperId','Publisher','Journal'])
    
    info = info.merge(paperYear, on='PaperId')

    acptDelay = pd.read_csv(PROJDIR + 'AcptDelay.csv', sep='\t', dtype={'AcptDelay':int})
    
    acptDelay = acptDelay[(acptDelay.AcptDelay > 0) & (acptDelay.AcptDelay <= 730)]

    journalAverage = (
        info.merge(acptDelay, on='PaperId').groupby(['Journal','Year'])
        .AcptDelay.mean().reset_index()
        .rename(columns={'AcptDelay':'JAvg'})
    )

    acptDelay = (
        acptDelay.merge(info, on='PaperId')
        .merge(journalAverage, on=['Journal','Year'])
    )
    
    return acptDelay


def loadPaperRecvAcptTime():
    
    recv = (
        pd.read_csv('/scratch/fl1092/followup-editors/RecvTime.csv', sep='\t', parse_dates=['RecvDate'])
        .assign(Year = lambda df: df.RecvDate.apply(lambda x: x.year))
    )

    acpt = (
        pd.read_csv('/scratch/fl1092/followup-editors/AcptTime.csv', sep='\t', parse_dates=['AcptDate'])
        .assign(Year = lambda df: df.AcptDate.apply(lambda x: x.year))
    )
    
    return recv, acpt