#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Team(object):

	"""docstring for Team"""
	def __init__(self,name,score,wins = 0,loses = 0,equals = 0):

		super(Team, self).__init__()
		self.name = name
		self.score = score
		self.wins = wins
		self.loses = loses
		self.equals = equals
		
	def teamWin(team):
		team.score  = team.score + 3
		team.wins = team.wins + 1
	def teamEqual(team):
		team.score = team.score + 1
		team.equals = team.equals + 1
	def teamLose(team):
		team.loses = team.loses + 1
		

def battle(team1,team2,result):
	if result == 0:
		team1.teamWin()
		team2.teamLose()
	elif result == 1:
		team1.teamEqual()
		team2.teamEqual()
	else:
		team1.teamLose()
		team2.teamWin()
chinaWin = 0
yiLang = Team('伊朗',11)
hanGuo = Team('韩国',10)
wuZi = Team('乌兹别克斯坦',9)
xuLiYa = Team('叙利亚',8)
china = Team('中国',5)
kaTaEr = Team('卡塔尔',4) 

def whoOutAisa():
	global chinaWin
	teams = [{'name':yiLang.name,'score':yiLang.score},
	{'name':hanGuo.name,'score':hanGuo.score},
	{'name':wuZi.name,'score':wuZi.score},
	{'name':xuLiYa.name,'score':xuLiYa.score},
	{'name':china.name,'score':china.score},
	{'name':kaTaEr.name,'score':kaTaEr.score}]
	a = sorted(teams,key = lambda x:x['score'],reverse = True)
	if a[0]['name'] == '中国' or a[1]['name'] == '中国':
		chinaWin = chinaWin + 1
		print(a,'\n')
count  = 0

for x in range(3):
	
	for a in range(3):
		
		for b in range(3):
			
			for c in range(3):
				
				for d in range(3):
					
					for e in range(3):
						
						for f in range(3):
							
							for g in range(3):
								
								for h in range(3):
									
									for i in range(3):
										
										for j in range(3):
											
											for k in range(3):
												battle(yiLang,china,j)
												battle(xuLiYa,china,i)
												battle(xuLiYa,kaTaEr,h)
												battle(kaTaEr,china,g)
												battle(wuZi,china,f)
												battle(hanGuo,yiLang,e)
												battle(hanGuo,wuZi,d)
												battle(hanGuo,kaTaEr,c)
												battle(yiLang,wuZi,b)
												battle(wuZi,kaTaEr,a)
												battle(hanGuo,xuLiYa,x)	
												battle(yiLang,xuLiYa,k)
												whoOutAisa()
												count =count + 1
												yiLang = Team('伊朗',11)
												hanGuo = Team('韩国',10)
												wuZi = Team('乌兹别克斯坦',9)
												xuLiYa = Team('叙利亚',8)
												china = Team('中国',5)
												kaTaEr = Team('卡塔尔',4) 
print('中国出现的概率为',chinaWin/count)













