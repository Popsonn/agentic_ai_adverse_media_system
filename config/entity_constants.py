# Entity Disambiguation Constants

ENTITY_CREDIBLE_SOURCES = [
    # PREMIUM NEWS & MEDIA (INTERNATIONAL + EMEA)
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnn.com", "bbc.com", "bbc.co.uk", "ap.org", "forbes.com", 
    "cnbc.com", "fortune.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "telegraph.co.uk", "independent.co.uk",
    "economist.com", "politico.eu", "euronews.com", "dw.com",
    "lemonde.fr", "spiegel.de", "corriere.it", "elpais.com",
    "handelsblatt.com", "lesechos.fr", "ilsole24ore.com",
    
    # US REGULATORY & GOVERNMENT
    "sec.gov", "justice.gov", "fbi.gov", "treasury.gov", "fincen.gov",
    "occ.gov", "federalreserve.gov", "cftc.gov", "finra.org",
    "gov.us",
    
    # UK/EU REGULATORY
    "fca.org.uk", "sfo.gov.uk", "gov.uk", "companieshouse.gov.uk",
    "europa.eu",
    
    # NIGERIAN REGULATORY & GOVERNMENT
    "cbn.gov.ng", "sec.gov.ng", "pencom.gov.ng", "naicom.gov.ng",
    "efcc.gov.ng", "icpc.gov.ng", "cac.gov.ng", "firs.gov.ng",
    "ndic.org.ng",
    
    # NIGERIAN FINANCIAL MARKETS
    "fmdqotc.com", "nse.com.ng", "nasd.com.ng",
    
    # NIGERIAN CREDIBLE MEDIA
    "punchng.com", "thenationonlineng.net", "guardian.ng",
    "vanguardngr.com", "dailytrust.com", "tribuneonlineng.com",
    "premiumtimesng.com", "sahareporters.com", "thecable.ng",
    "businessday.ng", "nairametrics.com", "pulse.ng",
    "thisday.ng", "thisdaylive.com", "channels.tv", "channelstv.com",
    "businessamlive.com", "leadership.ng", "newtelegraphng.com",
    "sunnewsonline.com", "independent.ng", "naijaentrepreneurs.com",
    
    # AFRICAN/EMEA MEDIA
    "mg.co.za", "businesslive.co.za", "fin24.com", "news24.com",
    "dailymaverick.co.za", "timeslive.co.za", "iol.co.za",
    "nation.co.ke", "standardmedia.co.ke", "businessdailyafrica.com",
    "theeastafrican.co.ke", "capitalfm.co.ke",
    "ghanaweb.com", "myjoyonline.com", "graphic.com.gh",
    "citinewsroom.com", "3news.com",
    "allafrica.com", "africanews.com", "theafricareport.com",
    "african.business", "ventures-africa.com",
    
    # INTERNATIONAL REGULATORY & EMEA
    "interpol.int", "fatf-gafi.org", "un.org", "worldbank.org",
    "opensanctions.org", "eba.europa.eu", "esma.europa.eu",
    "ecb.europa.eu", "bafin.de", "amf-france.org", "consob.it",
    "cnmv.es", "fma.gv.at", "cssf.lu", "mas.gov.sg",
    "sarb.co.za", "fsc.gov.za", "bou.or.ug",
    
    # COURT RECORDS & LEGAL
    "pacer.gov", "courtlistener.com", "justia.com",
    "lawnigeria.com", "nigerianlawguru.com",
    
    # INVESTIGATIVE JOURNALISM
    "icij.org", "occrp.org", "transparency.org", "propublica.org",
    
    # FINANCIAL/BANKING TRADE PUBLICATIONS
    "americanbanker.com", "bankingdive.com", "risk.net", 
    "complianceweek.com", "amlrightsource.com", "compliance.com",
    
    # BUSINESS INTELLIGENCE & DATABASES
    "crunchbase.com", "pitchbook.com", "bloomberg.law",
    "zoominfo.com", "apollo.io", "leadiq.com",
    
    # PROFESSIONAL NETWORKS
    "linkedin.com", "xing.com", "viadeo.com",
    
    # ACADEMIC & RESEARCH
    "wikipedia.org", "britannica.com", "researchgate.net",
    "scholar.google.com", "orcid.org", "academia.edu",
    "semanticscholar.org",
    
    # BIOGRAPHICAL & REFERENCE
    "biography.com", "whoswho.com", "marquisdirectory.com",
    
    # PROFESSIONAL ASSOCIATIONS
    "cfa.org", "aicpa.org", "rics.org", "bcs.org"
]

ENTITY_EXCLUDED_DOMAINS = [
    # SOCIAL MEDIA
    "facebook.com", "twitter.com", "instagram.com", 
    "tiktok.com", "youtube.com", "reddit.com", "pinterest.com",
    "snapchat.com", "whatsapp.com", "telegram.org",
    
    # USER-GENERATED CONTENT & FORUMS
    "quora.com", "yahoo.answers.com", "tumblr.com", "medium.com",
    "nairaland.com", "lindaikejisblog.com",  # Moved here - more forum/gossip
    
    # PROMOTIONAL/MARKETING SITES
    "pr.com", "prweb.com", "businesswire.com", "prnewswire.com",
    
    # SPAM/LOW QUALITY AGGREGATORS
    "scribd.com", "slideshare.net", "contactout.com",
    
    # DATING/PERSONAL
    "match.com", "tinder.com", "bumble.com",
    
    # CLASSIFIED/MARKETPLACE
    "craigslist.org", "ebay.com", "amazon.com/gp/profile",
    
    # FORUMS/COMMUNITIES
    "stackoverflow.com", "github.com/users",
    
    # ADULT/GAMING/ENTERTAINMENT
    "twitch.tv", "onlyfans.com", "patreon.com"
]