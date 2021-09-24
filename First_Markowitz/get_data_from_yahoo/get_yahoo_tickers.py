#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 23:23:28 2021

@author: leoprimero
"""


## With Pandas API

from pandas_datareader import data as pdr
import datetime as date
import yfinance as yf

def get_Data(index):
    data = pdr.get_data_yahoo(index,start=startdate,end=enddate)
    return data

startdate = date.datetime(2021, 9, 1)
enddate = date.datetime(2021, 9, 7)


## With Yahoo API

# def get_Data(ticker):
#     data = pdr.get_data_yahoo(index,start=startdate,end=enddate)
#     data = yf.download(ticker,start='2021-09-01', end='2021-09-07')
#     return data

# startdate = date.datetime(2020, 3, 24)
# enddate = date.datetime(2021, 7, 2)


########   Basic Materials		

NEM=get_Data("NEM")
GOLD=get_Data("GOLD")
AUY=get_Data("AUY")
HMY=get_Data("HMY")
SID=get_Data("SID")
GGB=get_Data("GGB")
X=get_Data("X")
FCX=get_Data("FCX")
CX=get_Data("CX")
DD=get_Data("DD")
BHP=get_Data("BHP")
RIO=get_Data("RIO")
VALE=get_Data("VALE")
PKX= get_Data("PKX")

NEMМОЙ = NEM.to_csv("NEM.csv")
GOLDМОЙ = GOLD.to_csv("GOLD.csv")
AUYМОЙ = AUY.to_csv("AUY.csv")
HMYМОЙ = HMY.to_csv("HMY.csv")
SIDМОЙ = SID.to_csv("SID.csv")
GGBМОЙ = GGB.to_csv("GGB.csv")
XМОЙ = X.to_csv("X.csv")
FCXМОЙ = FCX.to_csv("FCX.csv")
CXМОЙ = CX.to_csv("CX.csv")
DDМОЙ = DD.to_csv("DD.csv")
BHPМОЙ = BHP.to_csv("BHP.csv")
RIOМОЙ = RIO.to_csv("RIO.csv")
VALEМОЙ = VALE.to_csv("VALE.csv")
PKXМОЙ = VALE.to_csv("PKX.csv")


# 'NEM','GOLD','AUY','HMY','SID','GGB','X','FCX','CX','DD','BHP','RIO','VALE'
# rics = ['NEM',\
#         'GOLD',\
#         'AUY',\
#         'HMY',\
#         'SID',\
#         'GGB',\
#         'X',\
#         'FCX',\
#         'CX',\
#         'DD',\
#         'BHP',\
#         'RIO',\
#         'VALE']

# ########   Communication Services		
GOOGL=get_Data("GOOGL")
TWTR=get_Data("TWTR")
FB=get_Data("FB")
SNAP=get_Data("SNAP")
YELP=get_Data("YELP")
BIDU=get_Data("BIDU")
DIS=get_Data("DIS")
NFLX=get_Data("NFLX")
VZ=get_Data("VZ")
VOD=get_Data("VOD")
T=get_Data("T")
VIV=get_Data("VIV")

GOOGLМОЙ = GOOGL.to_csv("GOOGL.csv")
TWTRМОЙ = TWTR.to_csv("TWTR.csv")
FBМОЙ = FB.to_csv("FB.csv")
SNAPМОЙ = SNAP.to_csv("SNAP.csv")
YELPМОЙ = YELP.to_csv("YELP.csv")
BIDUМОЙ = BIDU.to_csv("BIDU.csv")
DISМОЙ = DIS.to_csv("DIS.csv")
NFLXМОЙ = NFLX.to_csv("NFLX.csv")
VZМОЙ = VZ.to_csv("VZ.csv")
VODМОЙ = VOD.to_csv("VOD.csv")
TМОЙ = T.to_csv("T.csv")
VIVМОЙ = VIV.to_csv("VIV.csv")


# 'GOOGL','TWTR','FB','SNAP','YELP','BIDU','DIS','NFLX','VZ','VOD','T','VIV'

# rics = ['GOOGL',\
#         'TWTR',\
#         'FB',\
#         'SNAP',\
#         'YELP',\
#         'BIDU',\
#         'DIS',\
#         'NFLX',\
#         'VZ',\
#         'VOD',\
#         'T',\
#         'VIV']


####     Consumer Cyclical		

MELI=get_Data("MELI")
EBAY=get_Data("EBAY")
AMZN=get_Data("AMZN")
JD=get_Data("JD")
BABA=get_Data("BABA")
SBUX=get_Data("SBUX")
MCD=get_Data("MCD")
ARCO=get_Data("ARCO")
TRIP=get_Data("TRIP")
DESP=get_Data("DESP")
TSLA=get_Data("TSLA")
TM=get_Data("TM")
HMC=get_Data("HMC")
NKE=get_Data("NKE")
LVS=get_Data("LVS")
HD=get_Data("HD")


MELIМОЙ = MELI.to_csv("MELI.csv")
EBAYМОЙ = EBAY.to_csv("EBAY.csv")
AMZNМОЙ = AMZN.to_csv("AMZN.csv")
JDМОЙ = JD.to_csv("JD.csv")
BABAМОЙ = BABA.to_csv("BABA.csv")
SBUXМОЙ = SBUX.to_csv("SBUX.csv")
MCDМОЙ = MCD.to_csv("MCD.csv")
ARCOМОЙ = ARCO.to_csv("ARCO.csv")
TRIPМОЙ = TRIP.to_csv("TRIP.csv")
DESPМОЙ = DESP.to_csv("DESP.csv")
TSLAМОЙ = TSLA.to_csv("TSLA.csv")
TMМОЙ = TM.to_csv("TM.csv")
HMCМОЙ = HMC.to_csv("HMC.csv")
NKEМОЙ = NKE.to_csv("NKE.csv")
LVSМОЙ = LVS.to_csv("LVS.csv")
HDМОЙ = HD.to_csv("HD.csv")

# 'MELI','EBAY','AMZN','JD','BABA','SBUX','MCD','ARCO','TRIP','DESP','TSLA','TM','HMC','NKE','LVS','HD'
# rics = ['MELI',\
#         'EBAY',\
#         'AMZN',\
#         'JD',\
#         'BABA',\
#         'SBUX',\
#         'MCD',\
#         'ARCO',\
#         'TRIP',\
#         'DESP',\
#         'TSLA',\
#         'TM',\
#         'HMC',\
#         'NKE',\
#         'LVS',\
#         'HD']
    
#####   Consumer Defensive		

PEP=get_Data('PEP')
FMX=get_Data('FMX')
KO=get_Data('KO')
ABEV=get_Data('ABEV')
KMB=get_Data('KMB')
PG=get_Data('PG')
CL=get_Data('CL')
# UN=get_Data('UN')
COST=get_Data('COST')
WMT=get_Data('WMT')
TGT=get_Data('TGT')
MO=get_Data('MO')
BRFS=get_Data('BRFS')
ADGO=get_Data('ADGO')
BG=get_Data("BG")


PEPМОЙ = PEP.to_csv("PEP.csv")
FMXМОЙ = FMX.to_csv("FMX.csv")
KOМОЙ = KO.to_csv("KO.csv")
ABEVМОЙ = ABEV.to_csv("ABEV.csv")
KMBМОЙ = KMB.to_csv("KMB.csv")
PGМОЙ = PG.to_csv("PG.csv")
CLМОЙ = CL.to_csv("CL.csv")
# UNМОЙ = UN.to_csv("UN.csv")
COSTМОЙ = COST.to_csv("COST.csv")
WMTМОЙ = WMT.to_csv("WMT.csv")
TGTМОЙ = TGT.to_csv("TGT.csv")
MOМОЙ = MO.to_csv("MO.csv")
BRFSМОЙ = BRFS.to_csv("BRFS.csv")
ADGOМОЙ = ADGO.to_csv("ADGO.csv")
BGМОЙ = ADGO.to_csv("BG.csv")

# 'PEP','FMX','KO','ABEV','KMB','PG','CL','UN','COST','WMT','TGT','MO','BRFS','ADGO'

# rics = ['PEP',\
#         'FMX',\
#         'KO',\
#         'ABEV',\
#         'KMB',\
#         'PG',\
#         'CL',\
#         'UN',\
#         'COST',\
#         'WMT',\
#         'TGT',\
#         'MO',\
#         'BRFS',\
#         'ADGO']


###   Energy		

PBR=get_Data('PBR')
XOM=get_Data('XOM')
BP=get_Data('BP')
CVX=get_Data('CVX')
TOT=get_Data('TOT')
SNP=get_Data('SNP')
PTR=get_Data('PTR')
SLB=get_Data('SLB')
UGP=get_Data('UGP')
VIST=get_Data('VIST')


PBRМОЙ =PBR.to_csv("PBR.csv")
XOMМОЙ =XOM.to_csv("XOM.csv")
BPМОЙ =BP.to_csv("BP.csv")
CVXМОЙ =CVX.to_csv("CVX.csv")
TOTМОЙ =TOT.to_csv("TOT.csv")
SNPМОЙ =SNP.to_csv("SNP.csv")
PTRМОЙ =PTR.to_csv("PTR.csv")
SLBМОЙ =SLB.to_csv("SLB.csv")
UGPМОЙ =UGP.to_csv("UGP.csv")
VISTМОЙ =VIST.to_csv("VIST.csv")

# 'PBR','XOM','BP','CVX','TOT','SNP','PTR','SLB','UGP','VIST'

# rics = ['PBR',\
#         'XOM',\
#         'BP',\
#         'CVX',\
#         'TOT',\
#         'SNP',\
#         'PTR',\
#         'SLB',\
#         'UGP',\
#         'VIST',\]

#####    Financial Services		

JPM=get_Data('JPM')
WFC=get_Data('WFC')
C=get_Data('C')
SAN=get_Data('SAN')
BCS=get_Data('BCS')
HSBC=get_Data('HSBC')
CS=get_Data('CS')
BBD=get_Data('BBD')
LYG=get_Data('LYG')
ITUB=get_Data('ITUB')
BSBR=get_Data('BSBR')
AXP=get_Data('AXP')
V=get_Data('V')
PYPL=get_Data('PYPL')
GS=get_Data('GS')
AIG=get_Data('AIG')

JPMМОЙ =JPM.to_csv("JPM.csv")
WFCМОЙ =WFC.to_csv("WFC.csv")
CМОЙ =C.to_csv("C.csv")
SANМОЙ =SAN.to_csv("SAN.csv")
BCSМОЙ =BCS.to_csv("BCS.csv")
HSBCМОЙ =HSBC.to_csv("HSBC.csv")
CSМОЙ =CS.to_csv("CS.csv")
BBDМОЙ =BBD.to_csv("BBD.csv")
LYGМОЙ =LYG.to_csv("LYG.csv")
ITUBМОЙ =ITUB.to_csv("ITUB.csv")
BSBRМОЙ =BSBR.to_csv("BSBR.csv")
AXPМОЙ =AXP.to_csv("AXP.csv")
VМОЙ =V.to_csv("V.csv")
PYPLМОЙ =PYPL.to_csv("PYPL.csv")
GSМОЙ =GS.to_csv("GS.csv")
AIGМОЙ =AIG.to_csv("AIG.csv")


# 'JPM','WFC','C','SAN','BCS','HSBC','CS','BBD','LYG','ITUB','BSBR','AXP','V','PYPL','GS','AIG'

# rics = ['JPM',\
#         'WFC',\
#         'C',\
#         'SAN',\
#         'BCS',\
#         'HSBC',\
#         'CS',\
#         'BBD',\
#         'LYG',\
#         'ITUB',\
#         'BSBR',\
#         'AXP',\
#         'V',\
#         'PYPL',\
#         'GS',\
#         'AIG']

#####    Healthcare		

PFE=get_Data('PFE')
JNJ=get_Data('JNJ')
GILD=get_Data('GILD')
MRK=get_Data('MRK')
BIIB=get_Data('BIIB')
BMY=get_Data('BMY')
AMGN=get_Data('AMGN')
GSK=get_Data('GSK')
NVS=get_Data('NVS')
ABT=get_Data('ABT')
MDT=get_Data('MDT')

PFEМОЙ =PFE.to_csv("PFE.csv")
JNJМОЙ =JNJ.to_csv("JNJ.csv")
GILDМОЙ =GILD.to_csv("GILD.csv")
MRKМОЙ =MRK.to_csv("MRK.csv")
BIIBМОЙ =BIIB.to_csv("BIIB.csv")
BMYМОЙ =BMY.to_csv("BMY.csv")
AMGNМОЙ =AMGN.to_csv("AMGN.csv")
GSKМОЙ =GSK.to_csv("GSK.csv")
NVSМОЙ =NVS.to_csv("NVS.csv")
ABTМОЙ =ABT.to_csv("ABT.csv")
MDTМОЙ =MDT.to_csv("MDT.csv")

# 'PFE','JNJ','GILD','MRK','BIIB','BMY','AMGN','GSK','NVS','ABT'

# rics = ['PFE',\
#         'JNJ',\
#         'GILD',\
#         'MRK',\
#         'BIIB',\
#         'BMY',\
#         'AMGN',\
#         'GSK',\
#         'NVS',\
#         'ABT']

######     Industrials		

GE=get_Data('GE')
MMM=get_Data('MMM')
HWM=get_Data('HWM')
LMT=get_Data('LMT')
BA=get_Data('BA')
RTX=get_Data('RTX')
ERJ=get_Data('ERJ')
CAT=get_Data('CAT')
DE=get_Data('DE')
FDX=get_Data('FDX')


GEМОЙ =GE.to_csv('GE.csv')
MMMМОЙ =MMM.to_csv('MMM.csv')
HWMМОЙ =HWM.to_csv('HWM.csv')
LMTМОЙ =LMT.to_csv('LMT.csv')
BAМОЙ =BA.to_csv('BA.csv')
RTXМОЙ =RTX.to_csv('RTX.csv')
ERJМОЙ =ERJ.to_csv('ERJ.csv')
CATМОЙ =CAT.to_csv('CAT.csv')
DEМОЙ =DE.to_csv('DE.csv')
FDXМОЙ =FDX.to_csv('FDX.csv')

# 'GE','MMM','HWM','LMT','BA','RTX','ERJ','CAT','DE','FDX'

# rics = ['GE',\
#         'MMM',\
#         'HWM',\
#         'LMT',\
#         'BA',\
#         'RTX',\
#         'ERJ',\
#         'CAT',\
#         'DE',\
#         'FDX']


######     Technology		

AMD=get_Data('AMD')
NVDA=get_Data('NVDA')
QCOM=get_Data('QCOM')
INTC=get_Data('INTC')
TXN=get_Data('TXN')
TSM=get_Data('TSM')
AAPL=get_Data('AAPL')
SONY=get_Data('SONY')
HPQ=get_Data('HPQ')
MSFT=get_Data('MSFT')
ADBE=get_Data('ADBE')
ORCL=get_Data('ORCL')
VRSN=get_Data('VRSN')
GLOB=get_Data('GLOB')
CRM=get_Data('CRM')
SAP=get_Data('SAP')
MSI=get_Data('MSI')
CSCO=get_Data('CSCO')
NOK=get_Data('NOK')
IBM=get_Data('IBM')
INFY=get_Data('INFY')
GRMN=get_Data('GRMN')
SQ=get_Data("SQ")



AMDМОЙ =AMD.to_csv('AMD.csv')
NVDAМОЙ =NVDA.to_csv('NVDA.csv')
QCOMМОЙ =QCOM.to_csv('QCOM.csv')
INTCМОЙ =INTC.to_csv('INTC.csv')
TXNМОЙ =TXN.to_csv('TXN.csv')
TSMМОЙ =TSM.to_csv('TSM.csv')
AAPLМОЙ =AAPL.to_csv('AAPL.csv')
SONYМОЙ =SONY.to_csv('SONY.csv')
HPQМОЙ =HPQ.to_csv('HPQ.csv')
MSFTМОЙ =MSFT.to_csv('MSFT.csv')
ADBEМОЙ =ADBE.to_csv('ADBE.csv')
ORCLМОЙ =ORCL.to_csv('ORCL.csv')
VRSNМОЙ =VRSN.to_csv('VRSN.csv')
GLOBМОЙ =GLOB.to_csv('GLOB.csv')
CRMМОЙ =CRM.to_csv('CRM.csv')
SAPМОЙ =SAP.to_csv('SAP.csv')
MSIМОЙ =MSI.to_csv('MSI.csv')
CSCOМОЙ =CSCO.to_csv('CSCO.csv')
NOKМОЙ =NOK.to_csv('NOK.csv')
IBMМОЙ =IBM.to_csv('IBM.csv')
INFYМОЙ =INFY.to_csv('INFY.csv')
GRMNМОЙ =GRMN.to_csv('GRMN.csv')
SQМОЙ =GRMN.to_csv('SQ.csv')

# 'AMD','NVDA','QCOM','INTC','TXN','TSM','AAPL','SONY','HPQ','MSFT','ADBE','ORCL','VRSN','GLOB','CRM','SAP','MSI','CSCO','NOK','IBM','INFY','GRMN'

# rics = ['AMD',\
#         'NVDA',\
#         'QCOM',\
#         'INTC',\
#         'TXN',\
#         'TSM',\
#         'AAPL',\
#         'SONY',\
#         'HPQ',\
#         'MSFT',\
#         'ADBE',\
#         'ORCL',\
#         'VRSN',\
#         'GLOB',\
#         'CRM',\
#         'SAP',\
#         'MSI',\
#         'CSCO',\
#         'NOK',\
#         'IBM',\
#         'INFY',\
#         'GRMN']

####   Utilities		

HNP=get_Data('HNP')
NGG=get_Data('NGG')
SBS=get_Data('SBS')

HNPМОЙ =HNP.to_csv('HNP.csv')
NGGМОЙ =NGG.to_csv('NGG.csv')
SBSМОЙ =SBS.to_csv('SBS.csv')

# 'HNP','NGG','SBS'

# rics = ['HNP',\
#         'NGG',\
#         'SBS']



SPY=get_Data('SPY')

SPYМОЙ =SPY.to_csv('SPY.csv')


###  Datos del Merval

ALUA=get_Data('ALUA.BA')
BBAR=get_Data('BBAR.BA')
BMA=get_Data('BMA.BA')
BYMA=get_Data('BYMA.BA')
CEPU=get_Data('CEPU.BA')
COME=get_Data('COME.BA')
CRES=get_Data('CRES.BA')
CVH	=get_Data('CVH.BA')
EDN	=get_Data('EDN.BA')
GGAL=get_Data('GGAL.BA')
HARG=get_Data('HARG.BA')
LOMA=get_Data('LOMA.BA')
MIRG=get_Data('MIRG.BA')
PAMP=get_Data('PAMP.BA')
SUPV=get_Data('SUPV.BA')
TECO2=get_Data('TECO2.BA')
TGNO4=get_Data('TGNO4.BA')
TGSU2=get_Data('TGSU2.BA')
TRAN=get_Data('TRAN.BA')
TXAR=get_Data('TXAR.BA')
VALO=get_Data('VALO.BA')
YPFD=get_Data('YPFD.BA')


ALUAМОЙ =ALUA.to_excel('ALUA.xlsx')
BMAМОЙ =BMA.to_excel('BMA.xlsx')
BYMAМОЙ =BYMA.to_excel('BYMA.xlsx')
CEPUМОЙ =CEPU.to_excel('CEPU.xlsx')
COMEМОЙ =COME.to_excel('COME.xlsx')
CRESМОЙ =CRES.to_excel('CRES.xlsx')
CVHМОЙ =CVH.to_excel('CVH.xlsx')
EDNМОЙ =EDN.to_excel('EDN.xlsx')
GGALМОЙ =GGAL.to_excel('GGAL.xlsx')
LOMAМОЙ =LOMA.to_excel('LOMA.xlsx')
MIRGМОЙ =MIRG.to_excel('MIRG.xlsx')
PAMPМОЙ =PAMP.to_excel('PAMP.xlsx')
SUPVМОЙ =SUPV.to_excel('SUPV.xlsx')
TECO2МОЙ =TECO2.to_excel('TECO2.xlsx')
TGNO4МОЙ =TGNO4.to_excel('TGNO4.xlsx')
TGSU2МОЙ =TGSU2.to_excel('TGSU2.xlsx')
TRANМОЙ =TRAN.to_excel('TRAN.xlsx')
VALOМОЙ =VALO.to_excel('VALO.xlsx')
YPFDМОЙ =YPFD.to_excel('YPFD.xlsx')





