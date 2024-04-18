import pandas as pd
PROJDIR = '/scratch/fl1092/COIpaper/'

def grapePaperAuthorEmb():
    
    # get author embedding by averaging paper embedding
    # paper embeddign calculated using paper-citation network
    
    embeddingPath = '/scratch/fl1092/COIpaper/expertise/grape/embedding/PaperReferenceGraphClean1Ite_central.csv'
    
    paperEmbedding = (
        pd.read_csv(embeddingPath, index_col=0)
        .reset_index().rename(columns={'index':'PaperId'})
    )
    
    authorEmbedding = (
        auJCount[['AuthorId']].drop_duplicates()

        .merge(papAu, on='AuthorId')
        .merge(paperEmbedding, on='PaperId')
        
        .groupby('AuthorId').agg({col: sum for col in [str(i) for i in range(100)]}).reset_index()
    )
    
    return authorEmbedding

def grapePaper(papAu, init=False):
    
    # paper-embedding calculated using paper citation network
    
    if init:
        editorVector = grapePaperAuthorEmb()
    else:
        editorVector = (
            pd.read_csv(PROJDIR + 'expertise/processed_emb/GrapePaperGraphAuthorEmbedding.csv')
            .assign(Embed=lambda df: list(df[[str(i) for i in range(100)]].to_numpy()) )
        )
        
    paperVector = getPaperVector(papAu=papAu, auVec=editorVector)
        
    return editorVector[['AuthorId','Embed']], paperVector

def getPaperVector(papAu, auVec):
    
    return (
        papAu.merge(auVec, on='AuthorId')
        .groupby('PaperId').Embed.mean().reset_index()
    )