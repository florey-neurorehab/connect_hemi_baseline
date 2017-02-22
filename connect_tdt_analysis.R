#Runs chi square + ANOVA + post hocs on Connect TDT sdata.
#Things of interest: Plot resids (fitted + hist + qq plot) to examine if ANOVA is "right fit"

library('tidyr')
library('foreign')
library('phia')
library('lmerTest')
library('psych')
library('multcomp') #For post-hoc comparisons of the main effects.
library('plyr')

ssdata<-read.csv('/home/peter/Desktop/Connect/connect_tdt_bl_culled.csv',header=TRUE, sep=',') #Load sdata
#sdata<-read.csv('/home/orcasha/Dropbox/post_doc/florey_leeanne/papers/Connect/connect_tdt_bl_culled.csv',header=TRUE, sep=',') #Load sdata
summary_d=describeBy(sdata,sdata$group) #Describe sdata by group, output as list of list for ease of viewing.
summary_m=describeBy(sdata,sdata$group,mat=TRUE) #Describe sdata by group, output as matrix for writing to spreadsheet.
write.csv(summary_m,file='/home/peter/Dropbox/post_doc/florey_leeanne/papers/Connect/baseline_hemi_paper/summary_d.csv',sep=",")
#write.csv(summary_m,file='/home/orcasha/Dropbox/post_doc/florey_leeanne/papers/Connect/baseline_hemi_paper/summary_d.csv',sep=",")
tbl=table(sdata$group, sdata$sex) #Create table of group x sex
print(tbl)
chisq.test(tbl,simulate.p.value=TRUE,B=10000) #Chi test @ 10,0000 reps

#Between groups t-tests
t.test(sdata[sdata$group=='c',]['age'],sdata[sdata$group=='l',]['age'],alternative=c('two.sided'),paired=FALSE, var.equal = FALSE)
t.test(sdata[sdata$group=='c',]['age'],sdata[sdata$group=='r',]['age'],alternative=c('two.sided'),paired=FALSE, var.equal = FALSE)
t.test(sdata[sdata$group=='l',]['post_stroke_days'],sdata[sdata$group=='r',]['post_stroke_days'],alternative=c('two.sided'),paired=FALSE, var.equal = FALSE)
t.test(sdata[sdata$group=='l',]['post_stroke_days']<1000,sdata[sdata$group=='r',]['post_stroke_days']<1000,alternative=c('two.sided'),paired=FALSE, var.equal = FALSE)
t.test(sdata[sdata$group=='l',]['lesion_size_vox'],sdata[sdata$group=='r',]['lesion_size_vox'],alternative=c('two.sided'),paired=FALSE, var.equal = FALSE)

###TDT###
sdata_long<-gather(sdata,cond,value,tdt_bl_aff:tdt_bl_unaff) #Create longform sdataset
sdata_long$cond=factor(sdata_long$cond) #VERY important to convert to factor post conversion...

#Recode groups for plotting
sdata_long$group <- recode(sdata_long$group, '"c" = "control"; "l" = "left"; "r" = "right"')

csdata <- ddply(sdata_long, c("group", "cond"), summarise, mean = mean(value), sd = sd(value), se = sd / sqrt(length(value)))

ggplot(sdata_long, aes(x = group, y = value, colour = cond)) + 
  geom_violin() + 
  theme_bw()

tdt.anova=lmer(value~group*cond+(1|no),sdata=sdata_long) #Run mixed anova y ~ f1 + f1 + f1*f2

tdt.means=interactionMeans(tdt.anova)
#plot(tdt.means)
anova(tdt.anova) #Generate p vals
summary(tdt.anova)

tdt.resid=(residuals(tdt.anova)) #Get resids for exploring appropriate use of parametric ANOVA

#Plot residuals & QQ
plot(fitted(tdt.anova),residuals(tdt.anova)) #Tests for independance
viewhist=hist(tdt.resid,prob=TRUE,breaks=20, xlab='TDT residuals') #Visualise normality of residuals
lines(density(tdt.resid),col='red',lwd=2) #Plots density curve
curve(dnorm(x,mean=mean(tdt.resid),sd=sd(tdt.resid)),col='blue',lty='dotted',lwd=2,add=TRUE) #Plots normal curve
#curve(dnorm(x,mean=6.628635e-15,sd=15.61253),col='blue',lty='dotted',lwd=2,add=TRUE) #Plots normal curve
legend(x=min(viewhist$breaks),y=max(viewhist$density)*.8,legend=c("Density curve","Normal curve"),col=c('red','blue'),lty=1:2)
qqnorm(tdt.resid)

#Run tests on interactions
sdata_long$GroupCond=interaction(sdata_long$group,sdata_long$cond) #Creates interaction term between conditions
tdt.anova2=lmer(value~GroupCond+(1|no),sdata=sdata_long) #Run mixed anova y ~ f1 + f1 + f1*f2

summary(glht(tdt.anova2,linfct=mcp(GroupCond='Tukey')))

# ###WPST###
# sdata_long<-gather(sdata,cond,value,wpst_bl_aff:wpst_bl_unaff) #Create longform sdataset
# sdata_long$cond=factor(sdata_long$cond) #VERY important to convert to factor post conversion...
# 
# wpst.anova=lmer(value~group*cond+(1|no),sdata=sdata_long) #Run mixed anova y ~ f1 + f1 + f1*f2
# 
# wpst.means=interactionMeans(wpst.anova)
# plot(wpst.means)
# anova(wpst.anova) #Generate p vals
# summary(wpst.anova)
# 
# wpst.resid=(residuals(wpst.anova)) #Get resids for exploring appropriate use of parametric ANOVA
# 
# #Plot residuals & QQ
# plot(fitted(wpst.anova),residuals(wpst.anova)) #Tests for independance
# viewhist=hist(wpst.resid,prob=TRUE,breaks=20, xlab='TDT residuals') #Visualise normality of residuals
# lines(density(wpst.resid),col='red',lwd=2) #Plots density curve
# curve(dnorm(x,mean=mean(wpst.resid),sd=sd(wpst.resid)),col='blue',lty='dotted',lwd=2,add=TRUE) #Plots normal curve
# legend(x=min(viewhist$breaks),y=max(viewhist$density)*.8,legend=c("Density curve","Normal curve"),col=c('red','blue'),lty=1:2)
# qqnorm(wpst.resid)
# 
# #Run tests on interactions
# sdata_long$GroupCond=interaction(sdata_long$group,sdata_long$cond) #Creates interaction term between conditions
# wpst.anova2=lmer(value~GroupCond+(1|no),sdata=sdata_long) #Run mixed anova y ~ f1 + f1 + f1*f2
# 
# summary(glht(wpst.anova2,linfct=mcp(GroupCond='Tukey')))
