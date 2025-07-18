import pandas as pd
import numpy as np
from datetime import datetime

def add_zeroes(frame):
    earliest = frame.index.min()
    latest = frame.index.max()
    date_range = pd.date_range(start=earliest, end=latest, freq='MS')
    zero_dates = date_range[np.isin(date_range, frame.index.to_numpy(), invert=True)]
    zero_dates = pd.DataFrame(np.zeros(len(zero_dates)), columns=['Count'],index=zero_dates)
    frame = pd.concat([frame, zero_dates]).sort_index()
    return frame

def open_kb_files(window_size):
    date_format = {True : '%Y/%m',  False: '%m/%Y'}
    def open_file(filename, yearfirst = False):
        df = pd.read_csv(r'Codefiles\KB_bestandstypen\\'+filename, 
                        names = ['ID', 'Date', 'File'])
        df['Frame'] = filename
        df['Date'] = pd.to_datetime(df['Date'], format = date_format[yearfirst])
        return df
    def open_cd_file(filename, yearfirst = False):
        df = pd.read_csv(r'Codefiles\KB_bestandstypen\\'+filename,
                        names = ['ID', 'Extension', 'Date'])
        df['Frame'] = filename
        return df
    BiomedOutput = open_file('BiomedOutput.csv')
    cbPubDateFilenameOutput = pd.read_csv(r'Codefiles\KB_bestandstypen\cbPubDateFilenameOutput.csv', 
                                         on_bad_lines='skip',
                                         names = ['ID', 'Date', 'File'])
    cbPubDateFilenameOutput['Date'] = pd.to_datetime(cbPubDateFilenameOutput['Date'], format = date_format[False])
    cbPubDateFilenameOutput['Frame'] = 'cbPubDateFilenameOutput'
    createdateAup = pd.read_csv(r'Codefiles\KB_bestandstypen\createdateAup.csv',
                                names = ['Date', 'File'])
    createdateAup['Date'] = pd.to_datetime(createdateAup['Date'], format = date_format[True])
    createdateAup['Frame'] = 'createdateAup'
    CrystolografiePubFileOutput = open_file('CrystolografiePubFileOutput.csv')
    
    DAREPubFile2Output = pd.read_csv(r'Codefiles\KB_bestandstypen\DAREPubFile2Output.csv',
                                     names = ['ID', 'Date', 'File'])
    DAREPubFile2Output['Date'] = pd.to_datetime(DAREPubFile2Output['Date'], format = date_format[False], errors='coerce')
    DAREPubFile2Output['Frame'] = 'DAREPubFile2Output'
    
    exportElsevierOutput = open_file('exportElsvierOutput.csv')
    exportSage2Output = open_file('exportSage2Output.csv')
    exportSpringer1Output = open_file('exportSpringer1Output.csv')
    exportSpringer2Output = open_file('exportSpringer2Output.csv')
    iospress1DateFilenameOutput = open_file('iospress1DateFilenameOutput.csv')
    iospress2DateFilenameOutput = open_file('iospress2DateFilenameOutput.csv')

    raapFilenameOutput = open_file('raapFilenameOutput.csv')
    SageDateFilenameOutput = open_file('SageDateFilenameOutput.csv')
    WebloketPeriodiekenDateFilenameOutput = open_file('WebloketPeriodiekenDateFilenameOutput.csv') 
    
    kb_metaframe = pd.concat([BiomedOutput, cbPubDateFilenameOutput, createdateAup, 
                              CrystolografiePubFileOutput, DAREPubFile2Output, exportElsevierOutput,
                              exportSage2Output, exportSpringer1Output, exportSpringer2Output, 
                              iospress1DateFilenameOutput, iospress2DateFilenameOutput, 
                              raapFilenameOutput, SageDateFilenameOutput, WebloketPeriodiekenDateFilenameOutput])
    kb_metaframe = kb_metaframe.drop_duplicates()
    
    kb_metaframe['Extension'] = kb_metaframe['File'].str.extract(r'\.([^.]+)$')
    kb_metaframe = kb_metaframe.drop(columns=['File'])
    kb_metaframe = kb_metaframe.loc[(kb_metaframe['Extension'] != '0') & (kb_metaframe['Extension']!='suppl')]
    kb_metaframe['Extension'] = kb_metaframe['Extension'].str.lower()
    kb_metaframe['Extension'] = kb_metaframe['Extension'].replace({f'{e}\n': e for e in ('doc', 'jpg', 'pdf', 'xls', 'xlsx', 'zip')})
    
    kb_metaframe.loc[kb_metaframe['Extension'].isin(['fa','faa','fasta', 'fas', 'fsa']), 'Extension'] = 'fa'
    kb_metaframe.loc[kb_metaframe['Extension'] == 'html', 'Extension'] = 'htm'
    kb_metaframe.loc[kb_metaframe['Extension'] == 'tiff', 'Extension'] = 'tif'
    kb_metaframe.loc[kb_metaframe['Extension'] == 'jpeg', 'Extension'] = 'jpg'
    kb_metaframe.loc[kb_metaframe['Extension'] == 'mpeg', 'Extension'] = 'mpg' 
    kb_metaframe.loc[kb_metaframe['Extension'].isin(['pdf', 'pdf-', 'pdf_', 'pdf_appendices', 'pdf_computational_simulations_of_second_language_construction_learning', 'pdf_sequence_1']), 'Extension'] = 'pdf'
    
    kb_metaframe['Date'] = pd.to_datetime(kb_metaframe['Date'].dt.strftime("%m-%Y"), format="%m-%Y")
    kb_metaframe = kb_metaframe.loc[kb_metaframe['Date'] <= datetime.today()]

    dareFileCreateDate = open_cd_file('dareFileCreateDate.output')
    elsevierCreateDate = open_cd_file('elsevierCreateDate.output')
    monoCreateDate = open_cd_file('monoCreateDate.output')    
    noKBMDOFileExtensieDate = open_cd_file('noKBMDOFileExtensieDate.output')

    dareFileCreateDate = dareFileCreateDate.loc[~dareFileCreateDate['Date'].isin(['1/1601', '12/1108'])]
    elsevierCreateDate = elsevierCreateDate.loc[~elsevierCreateDate['Date'].isin(['1/12', '1/18', '1/26', '1/9'])]
    monoCreateDate = monoCreateDate.loc[(monoCreateDate['Date'] != '1/101')]
    kb_cd_metaframe = pd.concat([dareFileCreateDate, elsevierCreateDate, monoCreateDate, noKBMDOFileExtensieDate])
    kb_cd_metaframe['ID'] = kb_cd_metaframe['ID'].apply(lambda x: x.removeprefix('./'))
    kb_cd_metaframe['Date'] = kb_cd_metaframe['Date'].apply(lambda x: x.rjust(7, '0'))
    kb_cd_metaframe['Date'] = pd.to_datetime(kb_cd_metaframe['Date'], format = '%m/%Y')
    kb_cd_metaframe['Date'].freq = 'MS'
    kb_cd_metaframe['Date'] = pd.to_datetime(kb_cd_metaframe['Date'])
    kb_cd_metaframe.drop_duplicates(inplace=True)
    kb_cd_metaframe.loc[kb_cd_metaframe['Extension']=='jpeg', 'Extension'] = 'jpg'
    kb_cd_metaframe.loc[kb_cd_metaframe['Extension']=='html', 'Extension'] = 'htm'
    kb_metaframe = pd.concat([kb_metaframe[['ID', 'Extension', 'Date']], kb_cd_metaframe[['ID', 'Extension', 'Date']]])
    del kb_cd_metaframe
    kb_metaframe.loc[kb_metaframe['Extension'].isin(['rdata', 'rda', 'rdat']), 'Extension' ] = 'rd'
    print(kb_metaframe.loc[kb_metaframe['Extension']=='rd'])
    kb_metaframe = kb_metaframe.groupby(by=['Extension','Date'], as_index = False).agg(Count=pd.NamedAgg(column='Date', aggfunc='count'))
    kb_metaframe.set_index('Date', inplace=True)
    kb_metaframe.index = pd.to_datetime(kb_metaframe.index)
    
    for ext, date in [('bin', '1981/08/12'), # bin 1981/08/12 https://www.ibm.com/history/personal-computer release datum eerste IBM pc
                    ('bmp', '1987/12/21'), # bmp 1987/12/21 https://microsoft.fandom.com/wiki/OS/2 eerste release op OS/2
                    ('cif', '1991'), # cif 1991, https://www.iucr.org/resources/cif
                    ('doc', '1985/09'),
                    ('docx', '2007/01/30'),
                    ('gif', '1987/06/15'), # gif 1987/06/15, https://www.nationalarchives.gov.uk/PRONOM/Format/proFormatSearch.aspx?status=detailReport&id=619
                    ('jpg', '1992'), # jpg 1992, https://jpeg.org/jpeg/
                    ('pdf', '1993/06/15'), # pdf 1993/06/15, https://www.adobe.com/acrobat/resources/pdf-timeline.html
                    ('ppt', '1987/04/20'), # ppt 1987/04/20, https://www.educba.com/powerpoint-version/
                    ('pptx', '2007/01/30'),
                    ('png', '1996/10/01'), # png 1996/10/01, https://www.w3.org/Graphics/PNG/
                    ('ps', '1984'), # ps 1984, https://www.prepressure.com/postscript/basics/history
                    ('py', '1989'), # py 1989, https://python-history.blogspot.com/2009/01/brief-timeline-of-python.html
                    ('r', '1993/08'), # r 1993/08, https://www.stat.auckland.ac.nz/~ihaka/downloads/Interface98.pdf
                    ('rar', '1995/04/22'), # rar 1995/04/22 http://www.oldversion.com/windows/winrar/
                    ('raw', '1987/12/21'), # raw 1987/12/21 https://microsoft.fandom.com/wiki/OS/2 eerste release op OS/2
                    ('sml', '2013/10/16'), # sml 2013/10/16 https://amateurphotographer.com/round-ups/camera_round_ups/10-years-of-sony-a7-the-camera-that-killed-the-dslr/ sml wordt gebruikt in 7-series camers aldus national archive https://www.nationalarchives.gov.uk/PRONOM/Format/proFormatSearch.aspx?status=detailReport&id=2616
                    ('svg', '2001/09/04'), # svg 2001/09/04 https://www.w3.org/TR/SVG10/
                    ('txt', '1963'), #https://everydaycomputeruser.blogspot.com/2020/11/plain-text.html
                    ('xml', '1998/01/01'), # xml 1998/01/01 https://www.nationalarchives.gov.uk/PRONOM/Format/proFormatSearch.aspx?status=detailReport&id=638
                    ('xlsx', '2007/01/30'),
                    ('zip', '1989/2/14')]:
        kb_metaframe = kb_metaframe.loc[~((kb_metaframe['Extension'] == ext) & 
                                          ((kb_metaframe.index < pd.to_datetime(pd.to_datetime(date), format='MS')) |
                                           (kb_metaframe.index > datetime.today())))]

    if window_size is not None:
        kb_metaframe = kb_metaframe.loc[(kb_metaframe['Extension'].groupby(kb_metaframe['Extension']).transform('size')>=window_size)]
        kb_metaframe = kb_metaframe.loc[kb_metaframe.groupby('Extension')["Count"].transform('nunique') > 1]
    return kb_metaframe


def open_dans_files(window_size):
    def open_dans_file(filename):
        df = pd.read_csv(r'Codefiles\bestandstypen\\'+filename, usecols= lambda x: x not in [], dtype={'path' : str,'contenttype' : str,'productiondate': str,'doi': str}) #'path', 'doi'
        return df

    arch = open_dans_file('arch-bestandstypen.csv')
    arch.loc[arch['productiondate'].str.contains('2023-11'), 'productiondate'] == '11-2023'
    arch = arch.loc[(arch['productiondate']!='1998') &
                    (arch['productiondate']!='1999') &
                    (arch['productiondate']!='2000') &
                    (arch['productiondate']!='2002') &
                    (arch['productiondate']!='2006') &
                    (arch['productiondate']!='2007') &
                    (arch['productiondate']!='2008') & 
                    (arch['productiondate']=='2009') &
                    (arch['productiondate']!='2010') & 
                    (arch['productiondate']!='2013') &
                    (arch['productiondate']!='2015') &
                    (arch['productiondate']!='2017') &
                    (arch['productiondate']!='2018') & 
                    (arch['productiondate']!='2019') & 
                    (arch['productiondate']!='2020') &
                    (arch['productiondate']!='2021') & 
                    (arch['productiondate']!='2024') ] 
    def arch_change_date(old_date, new_date):
            arch.loc[arch['productiondate']==old_date, 'productiondate'] = new_date

    for (old, new) in (('22-04-24', '22-04-2024'), # Verifieerbaar op https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/8AQPUU
                    ('21-04-22', '21-04-2022'), # Verifieerbaar op https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/9Z3TD3
                    ('2023-07', '07-2023'), ('2023-06', '06-2023'), ('2021-02', '02-2021'),
                    ('2022', '09-12-2022'),     # 09-12-2022 is deposit date: https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/7BQIKY
                    ('2022-6-20', '20-6-2022'), ('2023-05', '05-2023'), ('2023-1-13', '13-1-2023'), ('2014-1-7', '7-1-2014'), ('2023-04', '04-2023'), ('2020-08', '08-2020'),
                    ('2020-10', '10-2020'), ('2014-10-7', '7-10-2014'),
                    ('2023', '28-11-2023'), # 2023-11-28 is deposit date: https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/7XLGBU
                    ('2023-03', '03-2023'),
                    ('2004',  '14-06-2024'), # 2024-06-14 is deposit date: https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/GWUQYZ
                    ('2024-06', '06-2024'), ('2024-8-9', '9-8-2024'), ('2018-11', '11-2018'), ('2019-03', '03-2019'), ('2022-08', '08-2022'), ('2022-11-4', '4-11-2022'), ('2021-09', '09-2021'), ('2022-01', '01-2022'), ('21-04-23', '21-04-2023'), ('2023-02', '02-2023'), ('2022-07', '07-2022'), ('2021-01', '01-2021'), ('2023-3-20', '20-3-2023'), 
                    ('202-12-15', '15-12-2020'), # Verifieerbaar op https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/DB3RHF
                    ('2023-6-10', '10-6-2023'), ('2024-6-19', '19-6-2024'), ('2014-6-2', '2-6-2014'), ('2023-12-7', '7-12-2023'), ('2021-1-11', '11-1-2021'), ('2022-06', '06-2022'), ('2017-7-4', '4-7-2017'), ('2022-04', '04-2022'), ('2021-11', '11-2021'), ('2024-03', '03-2024'), ('2022-4-20',  '20-4-2022'), ('2009-12', '12-2009'), ('2016-02-4', '4-02-2016'), ('2023-6-29', '29-6-2023'), ('2023-3-9', '9-3-2023'), ('2023-11-1', '1-11-2023'), ('2022-03', '03-2022'), ('2017-07', '07-2017'), ('2023-01', '01-2023'), ('2022-10', '10-2022'),
                    ('2024-08', '08-2024'), ('2012-05', '05-2012'), ('2023-1-17', '17-1-2023'), ('2022-05', '05-2022'), ('2012-02', '02-2012'), ('2001-03', '03-2001'), ('2008-02', '02-2008'), ('2021-05', '05-2021'), ('2003-04', '04-2003'), ('2021-07', '07-2021'),
                    ('202-02-09', '09-02-2024'), # Verifieerbaar op https://archaeology.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/AR/DDDINZ
                    ('2024-4-29', '29-4-2024'), ('2012-07', '07-2012'), ('2023-10', '10-2023'), ('2007-04', '04-2007'),
                    ('2012-08', '08-2012'), ('202-05-29', '29-05-2024'), ('2003-02', '02-2003'), ('2022-7-8', '8-7-2022'),
                    ('2023-8-22', '22-8-2023'), ('2022-11', '11-2022'), ('2011-08-5', '5-08-2011'), ('2022-8-15', '15-8-2022'), ('2022-4-6', '6-4-2022'),
                    ('2023-5-12', '12-5-2023'), ('2019-12', '12-2019'), ('2022-02', '02-2022'), ('2022-11-2', '2-11-2022'), ('2023-9-5', '5-9-2023'), ('2012-04', '04-2012'),
                    ('2010-6-14', '14-6-2010'), ('2020-6-23', '23-6-2020'), ('2024-10-4', '4-10-2024'), ('2022-8-31', '31-8-2022'), ('2013-3-04', '04-3-2013'),
                    ('2001-9', '9-2001'), ('2023-8-7', '7-8-2023'), ('2022-7-12', '12-7-2022')):
            arch_change_date(old, new)
    
    ls = open_dans_file('ls-bestandstypen.csv')
    ls = ls.loc[~(ls['productiondate']=='2019')]
    pts = open_dans_file('pts-bestandstypen.csv')
    ssh = open_dans_file('ssh-bestandstypen.csv')
    ssh.loc[ssh['productiondate']=='2024-03', 'productiondate'] = '03-2024' 
    ssh.loc[ssh['productiondate']=='2024-10-1', 'productiondate'] = '1-10-2024'
    ssh.loc[ssh['productiondate']=='2015-11-4', 'productiondate'] = '4-11-2015'
    dataversenl = open_dans_file('bestandstypen-dataversenl.csv')
    dataversenl.loc[dataversenl['path'] == 'Data microarthropods Reijerscamp Sept 2012.xlsx', 'productiondate'] = '2012-09'
    dataversenl.loc[(dataversenl['path'].str.contains('July 2012')) & (dataversenl['productiondate'] == '2012'), 'productiondate'] = '2012-07'
    dataversenl.loc[dataversenl['path'] == 'Description database vitreous AGEs and DM 2016-11-15.docx', 'productiondate'] = '2016-11-15'
    dataversenl.loc[dataversenl['path'] == 'Database Maas et al Plos one - Risk factors radiographic progression AS (GLAS cohort) - anonymous 10-04-2017.sav', 'productiondate'] = '2017-04-10'
    dataversenl.loc[dataversenl['path'] == 'Ethical clearance Betty Tjipta Sari_Bilingualism... 17 Feb 2017.pdf', 'productiondate'] = '2017-02-17'
    dataversenl.loc[dataversenl['contenttype']=='"application/vnd.ms-excel"', 'contenttype'] = 'application/vnd.ms-excel'
    dataversenl = dataversenl.loc[dataversenl['productiondate'].str.len() > 4]
    
    dans_metaframe = pd.concat([arch, ls, pts, ssh, dataversenl]).sort_values(by=['productiondate', 'contenttype'])
    dans_metaframe.loc[dans_metaframe['contenttype']=='text/comma-separated-values', 'contenttype'] = 'text/csv'
    
    dans_metaframe = dans_metaframe.drop_duplicates()

    dans_metaframe['productiondate'] = pd.to_datetime(dans_metaframe['productiondate'], format='mixed', yearfirst=False)    
    dans_metaframe['productiondate'] = pd.to_datetime(dans_metaframe['productiondate'].dt.strftime('%m-%Y'), format='%m-%Y')
    dans_metaframe.index = pd.to_datetime(dans_metaframe.index)

    for ext, date in [('application/dbf', '1979'), # https://programminglanguages.info/language/dbase/
                    ('application/pdf', '1993/06/15'),
                    ('application/msword', '1983/10/25'), # https://microsoft.fandom.com/wiki/Microsoft_Word
                    ('application/octet-stream', '1977/04/16'), # De Apple II was de eerste PC die binary files als octet-stream kon openen https://historyofapple.com/computers/apple-ii/
                    ('application/prj', '1999/09'), # https://downloads.esri.com/support/whitepapers/other_/datamaps1999.pdf
                    ('application/rtf', '1987'), # https://mundobytes.com/nl/RTF-RTF-formaat/
                    ('application/sbn', '1998/02'), # https://oilit.com/HTML_Articles/1998_2_2.php
                    ('application/sbx', '1998/02'), # https://oilit.com/HTML_Articles/1998_2_2.php
                    ('application/shp', '1998/07'), # https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf
                    ('application/shx', '1998/07'), # https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf
                    ('application/x-sql', '1973'), # 
                    ('application/vnd.mif', '1994'), # Adobe kocht Frame in 1994 https://www.daube.ch/docu/files/david_murray.pdf
                    ('application/vnd.ms-excel', '1987/11/19'), # https://microsoft.fandom.com/wiki/Microsoft_Excel
                    ('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '2007/01/30'), # https://microsoft.fandom.com/wiki/Microsoft_Office_2007
                    ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '2007/01/30'), # https://microsoft.fandom.com/wiki/Microsoft_Office_2007
                    ('application/vnd.wordperfect', '1979'), # https://www.liquisearch.com/wordperfect/wordperfect_for_dos/history
                    ('application/x-msaccess', '1992/11/13'), # https://microsoft.fandom.com/wiki/Microsoft_Access
                    ('application/x-spss-por', '1968'), #https://www.loc.gov/preservation/digital/formats/fdd/fdd000468.shtml
                    ('application/x-spss-sav', '1996'), #https://www.loc.gov/preservation/digital/formats/fdd/fdd000469.shtml
                    ('application/x-spss-syntax', '1968'), # aanname
                    ('application/x-sql', '1986'), # https://docs.fileformat.com/database/sql/
                    ('application/x-stata', '1985/01'), # https://journals.sagepub.com/doi/epdf/10.1177/1536867X0500500102
                    ('application/x-tar', '1979/01'), # https://man.freebsd.org/cgi/man.cgi?query=tar&apropos=0&sektion=5&manpath=FreeBSD+7.0-RELEASE&arch=default&format=html
                    ('application/zip', '1989-2-14'), # http://cd.textfiles.com/pcmedic9310/MAIN/MISC/COMPRESS/ZIP.PRS
                    ('audio/midi', '1983'), # https://midi.org/midi-1-0
                    ('image/jpeg', '1992'),
                    ('image/tiff', '1986'), # https://www.coolutils.com/blog/what-is-tiff-file/
                    ('text/csv', '1972/07'), # https://bitsavers.trailing-edge.com/pdf/ibm/370/fortran/GC28-6884-0_IBM_FORTRAN_Program_Products_for_OS_and_CMS_General_Information_Jul72.pdf
                    ('text/html', '1993/06'), # https://www.w3.org/MarkUp/draft-ietf-iiir-html-01.txt
                    ('text/plain','1963'), #'https://everydaycomputeruser.blogspot.com/2020/11/plain-text.html'
                    ('text/xml', '1998/1/1'), # https://www.nationalarchives.gov.uk/PRONOM/fmt/101 
                    ('text/tab-separated-values', '1993/06'), #https://www.ac3filter.net/what-is-a-tsv-file/
                    ('text/tsv', '1993/06'), #https://www.ac3filter.net/what-is-a-tsv-file/
                    ('text/x-c', '1972'), # https://www.w3schools.com/c/c_intro.php
                    ('text/x-fixed-field', '1965/12') # https://smecc.org/library-resources/machine-readable%20cataloging%20marc%20by%20avram%5B1%5D.pdf
                    ]:
        dans_metaframe = dans_metaframe.loc[~((dans_metaframe['contenttype'] == ext) & 
                                              ((dans_metaframe['productiondate'] < pd.to_datetime(pd.to_datetime(date), format='MS')) |
                                               (dans_metaframe['productiondate'] > datetime.today())))]

    dans_metaframe.reset_index(inplace=True)
    MIME_Type_to_ext = pd.read_csv(r"Codefiles\mime_to_ext.csv")
    dans_metaframe = dans_metaframe.merge(MIME_Type_to_ext, how='left', left_on='contenttype', right_on='MIME-type')
    dans_metaframe.loc[dans_metaframe['path'] == 'HR Images Algerian steppic zone - Desertification status.7z', 'Extension'] = '7z'
    dans_metaframe.loc[(dans_metaframe['contenttype']=='text/x-fixed-field') & (dans_metaframe['path'].str.endswith('.dat')), 'Extension'] = 'dat'
    dans_metaframe.loc[(dans_metaframe['contenttype']=='text/x-fixed-field') & (dans_metaframe['path'].str.endswith('.asc')), 'Extension'] = 'asc' 

    octet_stream = (dans_metaframe['contenttype']=='application/octet-stream')
    for (end, ext) in [('.BAK', 'bak'),
                       ('.BAT', 'bat'),
                       ('.Bat', 'bat'),
                       ('.BK!', 'bk1'),
                       ('.CAT', 'cat'), 
                       ('.CPG','cpg'),
                       ('.DB', 'db'),
                       ('.DBF','dbf'),
                       ('.DS_Store', 'ds_store'),                        
                       ('.DDD', 'ddd'), 
                       ('.DLL', 'dll'), 
                       ('.EXE', 'exe'),
                       ('.F', 'f'),
                       ('.IMG', 'img'), 
                       ('.KMZ', 'kmz'), 
                       ('.LOG', 'log'),
                       ('.MAN', 'man'), 
                       ('.MD', 'md'),
                       ('.MID', 'mid'),
                       ('.MDX', 'mdx'),
                       ('.MOV', 'mov'), 
                       ('.MP3', 'mp3'),
                       ('.MUS', 'mus'), 
                       ('.PRG', 'prg'), 
                       ('.RTF', 'rtf'),
                       ('.Rd', 'rd'),
                       ('.rdata_part2', 'rd'),
                       ('.rdata_part3', 'rd'),
                       ('.rdata_part4', 'rd'),
                       ('.rdata_part5', 'rd'),
                       ('.rdata_part6', 'rd'),
                       ('.Rmd', 'rmd'),
                       ('.Rnw', 'rnw'), 
                       ('.SHP', 'shp'), 
                       ('.SHX', 'shx'),
                       ('.TTF', 'ttf'), 
                       ('.Txt', 'txt'), 
                       ('.TFW', 'tfw'),
                       ('.XSL', 'xsl'),
                       ('.Zip', 'zip'), 
                       ('.accdb', 'accdb'),
                       ('.bak', 'bak'), 
                       ('.bk1', 'bk1'),
                       ('.bk2', 'bk2'),
                       ('.bin', 'bin'),
                       ('.cdb', 'cdb'), 
                       ('.cfg', 'cfg'),
                       ('.cmd', 'cmd'),
                       ('.csv_part1', 'csv'),
                       ('.csv_part2', 'csv'),
                       ('.csv_part3', 'csv'),
                       ('.db', 'db'),
                       ('.dbt', 'dbt'),
                       ('.docx', 'docx'),
                       ('.ent', 'ent'),
                       ('.fa', 'fa'),
                       ('.faa', 'fa'),
                       ('.fasta', 'fa'),
                       ('.frq', 'frq'),
                       ('.hdr', 'hdr'),
                       ('.img', 'img'),
                       ('.lab', 'lab'), # Vermoedelijk dit: https://fileinfobase.com/extension/lab
                       ('.lis', 'lis'),
                       ('.map', 'map'), # https://www.nationalarchives.gov.uk/PRONOM/Format/proFormatSearch.aspx?status=detailReport&id=2797
                       ('.mdx', 'mdx'),
                       ('.mp4', 'mp4'),
                       ('.php', 'php'),
                       ('.pl', 'pl'),
                       ('.prg', 'prg'),
                       ('.spv', 'spv'),
                       ('.srn', 'srn'),
                       ('.tfw', 'tfw'),
                       ('.wp', 'wp'),
                       ('.xlsx', 'xlsx'),
                       ('.xsd', 'xsd')]:                       
        dans_metaframe.loc[octet_stream & (dans_metaframe['path'].str.endswith(end)), 'Extension'] = ext    
    dans_metaframe.loc[octet_stream & dans_metaframe['path'].str.startswith('asc/00000001'), 'Extension'] = 'asc'
    dans_metaframe.loc[octet_stream & (dans_metaframe['path'].str.endswith('.WP5') | dans_metaframe['path'].str.contains('wp5/00000001')), 'Extension'] = 'wp5'
    dans_metaframe.loc[octet_stream & ((dans_metaframe['path'] == 'Its4land_Audio-files_Rwanda/24. 39 mka') | 
                                       (dans_metaframe['path'] == 'Its4land_Audio-files_Kenya/5. 7mka')), 'Extension'] = 'mka'
    dans_metaframe.loc[octet_stream & (dans_metaframe['path'] == 'programs/laf-fabric/docs/_build/html/.buildinfo'), 'Extension'] = 'htm'
    dans_metaframe.loc[octet_stream & (dans_metaframe['path'].str.contains('gdb')), 'Extension'] = 'gdb'
    dans_metaframe = dans_metaframe.loc[(dans_metaframe['Extension']!='a')]
    dans_metaframe = dans_metaframe.drop_duplicates()
    dans_metaframe = dans_metaframe.drop(columns=['path', 'doi'])
    dans_metaframe = dans_metaframe.groupby(by=['Extension','productiondate'], as_index = False).agg(Count=pd.NamedAgg(column='productiondate', aggfunc='count'))
    dans_metaframe.set_index('productiondate', inplace=True)
    if window_size is not None:
        dans_metaframe = dans_metaframe.loc[(dans_metaframe['Extension'].groupby(dans_metaframe['Extension']).transform('size')>=window_size)]
        dans_metaframe = dans_metaframe.loc[dans_metaframe.groupby('Extension')['Count'].transform('nunique') > 1]
    return dans_metaframe

def open_na_files(window_size):
    na_metaframe = pd.read_csv(r'c:\users\jme060\Documents\Koninklijke Bibliotheek\bestandstypen\bestanden_edepot.csv', usecols=lambda x: x not in [], dtype={'FORMATPUID' : str,'FORMATNAME': str,'LASTMODIFIEDDATE': str,'DATE_': str,'FILEREF': str})
    na_metaframe.drop(columns=['FORMATNAME', 'DATE_', 'FILEREF'], inplace = True)
    PUID_to_ext = pd.read_csv(r'C:\Users\JME060\Documents\Koninklijke Bibliotheek\Codefiles\PUIDs naar extensie.csv')
    na_metaframe = na_metaframe.loc[na_metaframe['FORMATPUID'].isin(PUID_to_ext['PUID'].unique())]
    na_metaframe['LASTMODIFIEDDATE'] = na_metaframe['LASTMODIFIEDDATE'].map(lambda x: x[:10])
    na_metaframe = na_metaframe.drop_duplicates()
    na_metaframe['Date'] = pd.to_datetime(na_metaframe['LASTMODIFIEDDATE'], errors='coerce')
    na_metaframe = na_metaframe.drop(columns=['LASTMODIFIEDDATE'])      
    na_metaframe['Date'] = pd.to_datetime(na_metaframe['Date'].dt.strftime("%Y-%m"))
    PUIDinfo = pd.read_csv(r'C:\Users\JME060\Documents\Koninklijke Bibliotheek\Codefiles\PUIDinfo.csv')

    # Ensure Date is in datetime format
    PUIDinfo = PUIDinfo.dropna(subset=['PUID', 'ReleaseDate']).copy()
    PUIDinfo['ReleaseDate'] = pd.to_datetime(pd.to_datetime(PUIDinfo['ReleaseDate'], format='mixed'), format='MS')

    # Merge to get the ReleaseDate per FORMATPUID
    na_metaframe = na_metaframe.merge(PUID_to_ext[['PUID', 'ExternalSignature']], how='left', left_on='FORMATPUID', right_on='PUID')
    merged = na_metaframe.merge(PUIDinfo[['PUID', 'ReleaseDate']], how='left', left_on='FORMATPUID', right_on='PUID')
    
    # Filter: Keep only rows where Date is between ReleaseDate and today
    mask = (merged['Date'] >= merged['ReleaseDate']) & (merged['Date'] <= datetime.today())

    # Filtered frame
    na_metaframe = merged.loc[mask, ['ExternalSignature', 'Date']]

    na_metaframe.reset_index(inplace=True)
    na_metaframe = na_metaframe.groupby(by=['ExternalSignature', 'Date'], as_index = False).agg(Count=pd.NamedAgg(column='Date', aggfunc='count'))
    na_metaframe = na_metaframe.sort_values(['ExternalSignature', 'Date'])
    na_metaframe.set_index('Date', inplace=True)

    if window_size is not None:
        na_metaframe = na_metaframe.loc[(na_metaframe['ExternalSignature'].groupby(na_metaframe['ExternalSignature']).transform('size')>=window_size)]    
        na_metaframe = na_metaframe.loc[na_metaframe.groupby("ExternalSignature")["Count"].transform("nunique") > 1]
    na_metaframe.index = pd.to_datetime(na_metaframe.index)
    
    return na_metaframe

def open_kb_create_date_files(window_size):
    date_format = {True : '%Y/%m',  False: '%m/%Y'}
    def open_file(filename, yearfirst = False):
        df = pd.read_csv(r'c:\users\jme060\Documents\Koninklijke Bibliotheek\KB_bestandstypen\\'+filename, 
                        names = ['ID', 'Extension', 'Date'])
        df['Frame'] = filename
        return df
    dareFileCreateDate = open_file('dareFileCreateDate.output')
    elsevierCreateDate = open_file('elsevierCreateDate.output')
    monoCreateDate = open_file('monoCreateDate.output')    
    noKBMDOFileExtensieDate = open_file('noKBMDOFileExtensieDate.output')

    dareFileCreateDate = dareFileCreateDate.loc[~dareFileCreateDate['Date'].isin(['1/1601', '12/1108'])]
    elsevierCreateDate = elsevierCreateDate.loc[~elsevierCreateDate['Date'].isin(['1/12', '1/18', '1/26', '1/9'])]
    monoCreateDate = monoCreateDate.loc[(monoCreateDate['Date'] != '1/101')]
    kb_cd_metaframe = pd.concat([dareFileCreateDate, elsevierCreateDate, monoCreateDate, noKBMDOFileExtensieDate])
    kb_cd_metaframe['ID'] = kb_cd_metaframe['ID'].apply(lambda x: x.removeprefix('./'))
    kb_cd_metaframe.set_index('ID', inplace=True)
    kb_cd_metaframe['Date'] = kb_cd_metaframe['Date'].apply(lambda x: x.rjust(7, '0'))
    kb_cd_metaframe['Date'] = pd.to_datetime(kb_cd_metaframe['Date'], format = '%m/%Y')
    kb_cd_metaframe['Date'].freq = 'MS'
    kb_cd_metaframe['Date'] = pd.to_datetime(kb_cd_metaframe['Date'])
    for ext, date in [('doc', '1985/09'),
                      ('docx', '2007/01/30'),
                      ('jpg', '1992'),
                      ('pdf', '1993/06/15'),
                      ('xlsx', '2007/01/30'),
                      ('zip', '1989/2/14')]:
        kb_cd_metaframe = kb_cd_metaframe.loc[~((kb_cd_metaframe['Extension'] == ext) & 
                                                ((kb_cd_metaframe['Date'] < pd.to_datetime(pd.to_datetime(date), format='MS')) |
                                                 (kb_cd_metaframe['Date'] > datetime.today())))]
    kb_cd_metaframe.drop_duplicates(inplace=True)
    kb_cd_metaframe = kb_cd_metaframe.groupby(by=['Extension','Date'], as_index = False).agg(Count=pd.NamedAgg(column='Date', aggfunc='count'))
    kb_cd_metaframe.reset_index(inplace=True)
    kb_cd_metaframe.drop(columns=['index'], inplace=True)
    kb_cd_metaframe.set_index('Date', inplace=True)
    if window_size is not None:
        kb_cd_metaframe = kb_cd_metaframe.loc[(kb_cd_metaframe['Extension'].groupby(kb_cd_metaframe['Extension']).transform('size')>=window_size)]
        kb_cd_metaframe = kb_cd_metaframe.loc[kb_cd_metaframe.groupby('Extension')["Count"].transform('nunique') > 1]

    return kb_cd_metaframe

def open_all_files(window_size):
    kb_metaframe = open_kb_files(None)
    kb_metaframe.reset_index(inplace=True)
    dans_metaframe = open_dans_files(None) 
    dans_metaframe.reset_index(inplace=True)
    dans_metaframe.rename(columns={'productiondate': 'Date'}, inplace=True) 
    na_metaframe = open_na_files(None) 
    na_metaframe.rename(columns={'ExternalSignature': 'Extension'}, inplace=True)
    na_metaframe.reset_index(inplace=True)
    all_metaframe = pd.concat([kb_metaframe, dans_metaframe, na_metaframe]) #, kb_cd_metaframe 
    all_metaframe = all_metaframe.groupby(['Extension', 'Date'], as_index=False).agg('sum').set_index('Date')
    all_metaframe = all_metaframe.loc[(all_metaframe['Extension'].groupby(all_metaframe['Extension']).transform('size')>=window_size)]
    all_metaframe = all_metaframe.loc[all_metaframe.groupby('Extension')['Count'].transform('nunique') > 1]
    return all_metaframe