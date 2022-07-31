def html_scrape(url):
    '''
    Just an example for *static* website-scraping,
    If websites include contents rendered by JavaScripts,
    Selenuim or requests_html are recommended.
    '''
    import requests
    import BeautifulSoup    
    headers = {
        'User-Agent': 'Mozilla/5.0',  # pretend I am a browser
        }
    session = requests.Session() #setup session
    data = session.get(url, headers=headers) #scrape the data
    soup = BeautifulSoup(data.text, 'html.parser') #parse the data
    return soup #return the parsed data

def js_scrape(url, headers):
    '''
    Scraping by requests_html, applicable in Jupyter Notebook
    
    # Since Jupyterhub runs an underlying async function,
    # 'await' is necessary if your website needs to send multiple requests
    # while waiting for the full contents to be rendered
    '''
    
    # execute the codes as below
    '''
    from requests_html import AsyncHTMLSession 
    asession = AsyncHTMLSession()
    r = await asession.get(url=url, headers=headers)
    await r.html.arender()
    resp=r.html.raw_html
    return resp
    '''
    