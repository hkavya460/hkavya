#binomial distribution is disceret 
#The binomial distribution is a discrete probability distribution of x successes in a sequence of n Bernoulli trials, each with a probability of success p. 

#properties ofbinomial distribution 
#There should be a finite number of trials
    #The n trials are considered to be independent of each other
#    Each trial should have only two mutually exclusive outcomes
 #   The probability of success p remains same from trial to trial

#nCx=n!r!(n−r)! * p * (1-p)^n-x 


#The plot of binomial probability distribution

#The discrete probability values Pb(x,n,p)
# p  is less than 0.5, the distribution is skewed to the left 

#Let x be the number of successes in n Bernoulli trials, each with a probability of success p.


n = 10   ## number of trials
p = 0.5  ## probability of success in a trial
x = 3    ## number of successes
#dbinom(x,n,p) returns the binomial probability for x successes in n trials,
binomial_probability = dbinom(x,n,p)
#########################################################

### pbinom(x,n,p) gives cumulative probability for a binomial distribution from 0 to x. 
cumulative_probability = pbinom(x,n,p)
print(paste("cumulative binomial probability for (",x,",",n,",",p,") = ",cumulative_probability))
#######################################################


### Function for finding x value corresponing to a cumulative probability
x_value = qbinom(cumulative_probability, n, p)


### Function that returns 4 random deviates from a Binomial distribution of given (n,p)
deviates = rbinom(4, n, p)
print(paste("4 binomial deviates :  "))
print(deviates)

par(mfrow = c(2,1))

### plot a binomial density distribution using dbinom()
n = 10
p = 0.4
x = seq(0,10)
pdens = dbinom(x,n,p)
plot(x,pdens, type="h", col="red", xlab = "Binomial variable x", ylab="binomial probability", 
       main="binomial probability distribution")

# We generate frequency histogram of binomial deviates using rbinom()
n = 10
p = 0.5
xdev= rbinom(10000, n, p)
plot(table(xdev), type="h", xlab="binomial variable x", ylab="frequency",
            main="frequency distribution of binomial random deviates")


