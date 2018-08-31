library(XML)
library(tidyr)
library(ggplot2)
library(lubridate)
library(gridExtra)

web <- 'http://fedearroz.com.co/new/precios.php'
arroz <- readHTMLTable(web,
                       header = T,
                       which = 1,
                       stringAsFactors = F)
#Data Wrangling

arroz[1] <- NULL
arroz$Month <- seq(1:12)

ld <- gather(data = arroz, 
             key = 'Year', 
             value = 'Price', 
             -c(Month))
ld$Year <- as.numeric(ld$Year)
ld$Date <- as.Date(as.character(ld$Year*10000+ld$Month*100+1),format = '%Y%m%d')
ld$Price <- gsub(',', '', ld$Price)
ld$Price <- as.numeric(ld$Price) 
ld$Price <- ld$Price/1000 #Convert Price from Tons to Kilo
ld <- ld[, c(4,3,1,2)] #Reorder Data frame
ld <- ld[, -c(3:4)] # delete columns 3 through 4
ld <- drop_na(ld) #Drop NA Values

#Plot Time Series and Histogram

p1 <- ggplot(data = ld, aes(x = Date, y = Price)) +
  geom_line(color = 'blue', size = 0.8) +
  ggtitle("Rice Price Fedearroz")

p2 <- qplot(ld$Price,
            geom="histogram",
            binwidth = 10,  
            main = "Rice Price Distribution", 
            xlab = "Price",  
            fill=I("blue"), 
            col=I("red"), 
            alpha=I(.2))

grid.arrange(p1, p2, nrow = 1)


x <- ld$Price
y = ts(x,frequency=12, 
       start = c(year(ld$Date[1]),month(ld$Date[1])), 
       end = c(year(ld$Date[length(ld$Date)]),month(ld$Date[length(ld$Date)])))
