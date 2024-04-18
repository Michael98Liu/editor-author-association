import glob
import pandas as pd

PROJDIR = '/scratch/fl1092/COIpaper/' # directory for the COI paper specifically
MAGDIR = '/scratch/fl1092/MAG/2021-12-06/'

# !pip install grape

auJSubset = (
    pd.read_csv(PROJDIR + "expertise/AuthorJournalCount.csv")
)

journals = auJSubset[['JournalId']].drop_duplicates()
authors = auJSubset[['AuthorId']].drop_duplicates()


reference = pd.read_csv(MAGDIR+"mag/PaperReferences.txt", sep="\t",
                       names = ['CitesFrom', 'BeingCited'], memory_map=True)

papAu = pd.read_csv(MAGDIR+"mag/PaperAuthorAffiliations.txt", sep="\t",
                      names = ['PaperId', 'AuthorId', 'AffiliationId', 'AuthorSequenceNumber', 
                               'OriginalAuthor', 'OriginalAffiliation'],
                      usecols = ['PaperId', 'AuthorId'], 
                    dtype = {'PaperId':int, 'AuthorId':int, 'AffiliationId':float}, memory_map=True).drop_duplicates()

papAuSubset = papAu.merge(authors, on='AuthorId')

papSubset = papAuSubset[['PaperId']].drop_duplicates()


# reduce memory consumption #
papAu = None
papAuSubset = None
authors = None
auJSubset = None
# reduce memory consumption #


reference = (
    reference
    
    .merge(papSubset.rename(columns={'PaperId':'CitesFrom'}).assign(A=True), on='CitesFrom', how='left')
    .merge(papSubset.rename(columns={'PaperId':'BeingCited'}).assign(B=True), on='BeingCited', how='left')
    
    .fillna({'A': False, 'B': False})
    .assign(Keep=lambda df: df.A + df.B)
)

reference = reference.query('Keep==True')

reference.drop(['A','B','Keep'], axis=1).to_csv(PROJDIR + 'expertise/grape/PaperReferenceGraphClean.csv',index=False)