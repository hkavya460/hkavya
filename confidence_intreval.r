
#calculate confidence interval for a given data set

data = c(0.360,1.827,0.372,0.610,0.521,1.18,0.534,0.898,0.319,0.63,0.614,0.374,0.411,0.406,0.533,0.788,0.449,0.348,0.413,0.662,0.273,
	0.262,1.925,0.767,1.177,2.46,0.448,0.55,0.385,0.307,0.57,0.97,0.622,0.674,1.499)
print(length(data))
var = 0.36
sigma = 0.6 
mean = mean(data)
print("mean is ",)
print(mean)

confidence_interval = function(alpha,mean,sigma)
{

z_score = qnorm((1- alpha )/ 2)
i = z_score * (sigma/ sqrt(length(data)))
upper_interval = mean + i 
lower_interval = mean - i 
print(paste("mu  for" ,alpha ,"lies between" ,  lower_interval , "and", upper_interval)) 
}
confidence_interval(0.99,mean,sigma)
confidence_interval(0.95,mean,sigma)
confidence_interval(0.68,mean,sigma)






