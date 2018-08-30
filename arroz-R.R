library(XML)
library(tidyr)
library(reshape2)

web <- 'http://fedearroz.com.co/new/precios.php'
arroz <- readHTMLTable(web,
                       header = T,
                       which = 1,
                       stringAsFactors = F)
arroz[1] <- NULL
arroz$Month <- seq(1:12)


ld <- gather(data = arroz, 
             key = 'Year', 
             value = 'Price', 
             -c(Month))
ld$Date <- ld$Year*10000+ld$Month*100+1
