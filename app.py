from flask import Flask, render_template, redirect, url_for, request
import math
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from itertools import product
import math
from math import pi
import pandas as pd
import time
import os
import torch

STEP_SIZE = 0.1

def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = ''
    df2['_tmpkey'] = ''

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res

def getVolume(l,d,di,nh,a,p,n, br, STEP_SIZE):
    vols = []
    maxU = (d-di)/2
#     time_to_burn = (r-ri)/br
    i = 0
    u = 2 * br * i
    vol = pi/4 * ( (d-u)**2 - nh*(di+u)**2) * (l - u)
    
    while(u < maxU and vol>0):
        vol = pi/4 * ( (d-u)**2 - nh*(di+u)**2) * (l - u)
        vols.append(vol)
        i += STEP_SIZE 
        u = 2 * br * i
    return vols

def getSA_AB_P(l,d,di,nh,a,p,n,dt, cstar, STEP_SIZE):
    #OLD SA CODE
    # sas = []
    # maxU = (d-di)/2
    # i = 0
    # u = 2 * br * i
    # while(u < maxU):
    #     sa = pi * ( ( (d-u) + nh*(di + u)) * (l - u) + ((d-u)**2 - nh*(di+u)**2)/2)
    #     sas.append(sa)
    #     i += STEP_SIZE 
    #     u = 2 * br * i
    
    #NEW SA CODE AUG 2023
    sa_list = []
    ab_list = []
    p_list = []
    
    #calculated
    at = math.pi * dt**2 /4
    rho = 1
    
    #initial values
    sa = math.pi * d**2 /4 * l + math.pi * di**2 /4 * l * nh #initial surface area
    ab = 0 #area burnt
    x = STEP_SIZE
    saCurr= sa
    
    i = 0
    while ab<sa or i<100:
        abCurr = nh * pi * (
            ((di + 2*x) * (l - 2*x)) + 
            ((d - 2*x) * (l - 2*x)) + 
            0.5 * ((d - 2*x)**2) - ((di + 2*x)**2)
        )
        
        if abCurr < 0:
            break
        
        print(abCurr)
        saCurr = saCurr - abCurr
        pCurr = abCurr/at * (a*rho*cstar/9800) ** (1/(1-n))
        
        sa_list.append(saCurr)
        ab_list.append(abCurr)
        p_list.append(pCurr)
        
        x += x
        i+=1 
        
    print("HI THERE THIS IS THE SA LIST")
    print(sa_list)
        
    return sa_list, ab_list, p_list


app = Flask(__name__)

# @app.route('/<name>')
# def index(name):
#     return render_template("index.html", boy = name)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/report', methods=['POST', 'GET'])
def report():
    if request.method == 'POST':
        
        D = float(request.form.get('D'))
        d = float(request.form.get('d'))
        n = int(request.form.get('n')) #num of holes
        L = float(request.form.get('L'))
        z=  float(request.form.get('z'))
        a = float(request.form.get('a'))
        p = float(request.form.get('p'))
        N = float(request.form.get('N'))
        dt = float(request.form.get('dt'))
        cstar = float(request.form.get('cstar'))
        report = request.form

        vols=[]
        sas=[]
        steps = []
        br = a*p**N
        maxU=(D-d)/2
        i=0
        u=2*br*i
        vol= math.pi/4 * ( (D-u)**2 - n*(d+u)**2) * (L - u)
        while(u < maxU and vol>0):
            vol = math.pi/4 * ( (D-u)**2 - n*(d+u)**2) * (L - u)
            vols.append(float(vol))
            
            sa = math.pi * ( ( (D-u) + n*(d + u)) * (L - u) + ((D-u)**2 - n*(d+u)**2)/2)
            sas.append(sa)
            
            i += z 
            u = 2 * br * i
            steps.append(i)
            
        sa_list, ab_list, p_list = getSA_AB_P(L,D,d,n,a,p,N,dt, cstar, STEP_SIZE)
        vol_list = getVolume(L,D,d,n,a,p,N, br, STEP_SIZE)
                       
        plt.figure()
        plt.plot(steps, vols)
        plt.title("Vols vs Time")
        plt.xlabel("Time")
        plt.ylabel("Volume")
        # plt.legend()
        plt.savefig('static/images/Volume.png')
        
        plt.figure()
        plt.plot(steps, sas)
        plt.title("Surface Area vs Time")
        plt.xlabel("Time")
        plt.ylabel("Surface Area")
        # plt.legend()
        plt.savefig('static/images/Surface Area.png')
        
        os.chdir(os.getcwd())
        figs = os.listdir('static/images')
        figs = ['images/' + file for file in figs]
        
        # return redirect(url_for('showVar', varName = D))  
        
        sns.reset_defaults()
        sns.set(
            rc={'figure.figsize':(7,5)}, 
            style="white" # nicer layout
        )
        
        fig = sns.lineplot(x=steps, y = vols).get_figure()
        fig.savefig("del.png")
       
        
        return render_template("report.html", report=report, figs = figs)
    
    
    
    
    
    
    else:
        D = request.args.get('D')
        return render_template("index.html")
        # return redirect(url_for('showVar', varName =D))

@app.route('/createDB')
def createDB():
    return render_template("createDB.html")

@app.route('/databaseReport',methods=['POST', 'GET'])
def databaseReport():
    if request.method == "POST":
        report = request.form
        Dstart  = float(report['Dstart'])
        Dend    = float(report['Dend'])
        Dstep   = float(report['Dstep'])
        Distart  = float(report['Distart'])
        Diend    = float(report['Diend'])
        Distep   = float(report['Distep'])
        Gstart  = float(report['Gstart'])
        Gend    = float(report['Gend'])
        Gstep   = float(report['Gstep'])
        astart  = float(report['astart'])
        aend    = float(report['aend'])
        astep   = float(report['astep'])
        Pstart  = float(report['Pstart'])
        Pend    = float(report['Pend'])
        Pstep   = float(report['Pstep'])
        Nstart  = float(report['Nstart'])
        Nend    = float(report['Nend'])
        Nstep   = float(report['Nstep'])
        Nhstart  = float(report['Nhstart'])
        Nhend    = float(report['Nhend'])
        Nhstep   = float(report['Nhstep'])

        #HARDCODED BIT HERE
        VolStep = 0.01
        SAStep = 0.01
        
        
        # d = np.arange(Dstart, Dend, Dstep)
        # l = np.arange(Gstart, Gend, Gstep)
        # di = np.arange(Distart, Diend, Distep)
        # nh = np.arange(Nhstart, Nhend, Nhstep) #number of holes
        # a = np.arange(astart, aend, astep)
        # p = np.arange(Pstart, Pend, Pstep)
        # n = np.arange(Nstart, Nend, Nstep)
              
        #DEMO CODE WITH ANY INPUT IS HERE
        
        d = np.arange(2,10,4)
        di = np.arange(1,8,5)
        l= np.arange(2,100,50)
        nh = np.arange(1,39, 20) #number of holes
        a = np.arange(1,5,3)
        p = np.arange(20,100,40)
        n = np.arange(0,1,0.4)
        cstar = np.arange(1,5,3)
        dt = np.arange(1,5,3)

        dfx = df_crossjoin(pd.DataFrame(l, columns = ['l']), pd.DataFrame(d, columns = ['d'])).droplevel(1)
        dfx = df_crossjoin(dfx, pd.DataFrame(di, columns = ['di'])).droplevel(1)
        dfx = df_crossjoin(dfx, pd.DataFrame(nh, columns = ['nh'])).droplevel(1)
        dfx = df_crossjoin(dfx, pd.DataFrame(a, columns = ['a'])).droplevel(1)

        dfx.to_csv("temp.csv", index=False)

        # print(dfx.head())

        df_iterator = pd.read_csv('temp.csv', 
                            dtype={'l' : float, 'd':float, 'di':float, 'nh':float, 'a':float}, 
                            chunksize=1e6, 
                            # header = None, 
                            on_bad_lines='skip')

        for i, df in enumerate(df_iterator):
            dfx = df_crossjoin(df, pd.DataFrame(p, columns = ['p'])).droplevel(1)
            mode = 'w' if i == 0 else 'a' 
            header = i == 0
            dfx.columns = ['l','d','di', 'nh', 'a', 'p']
            dfx[1:].to_csv("temp.csv", 
                    mode= mode,
                    header = header, 
                    index = False)
            
            
         #THIS IS ONE CHUNK
        df_iterator = pd.read_csv('temp.csv', 
                            dtype={'l' : float, 'd':float, 'di':float, 'nh':float, 'a':float, 'p':float}, 
                            chunksize=1e6, 
                            # header = None, 
                            on_bad_lines='skip')
        
       
        for i, df in enumerate(df_iterator):
            dfx = df_crossjoin(df, pd.DataFrame(n, columns = ['n'])).droplevel(1)
            mode = 'w' if i == 0 else 'a' 
            header = i == 0
            dfx.columns = ['l','d','di', 'nh', 'a', 'p', 'n']
            dfx[1:].to_csv("temp.csv", 
                    mode= mode,
                    header = header, 
                    index = False)
        #CHUNK ENDS
        #chunk starts
        df_iterator = pd.read_csv('temp.csv', 
                            dtype={'l' : float, 'd':float, 'di':float, 'nh':float, 'a':float, 'p':float, 'n':float}, 
                            chunksize=1e6, 
                            # header = None, 
                            on_bad_lines='skip')

        for i, df in enumerate(df_iterator):
            dfx = df_crossjoin(df, pd.DataFrame(cstar, columns = ['cstar'])).droplevel(1)
            mode = 'w' if i == 0 else 'a' 
            header = i == 0
            dfx.columns = ['l','d','di', 'nh', 'a', 'p', 'n', 'cstar']
            dfx[1:].to_csv("temp.csv", 
                    mode= mode,
                    header = header, 
                    index = False)
            
        #chunk ends
         #THIS IS ONE CHUNK
        df_iterator = pd.read_csv('temp.csv', 
                            dtype={'l' : float, 'd':float, 'di':float, 'nh':float, 'a':float, 'p':float, 'n':float, 'cstar':float}, 
                            chunksize=1e6, 
                            # header = None, 
                            on_bad_lines='skip')
        
       
        for i, df in enumerate(df_iterator):
            dfx = df_crossjoin(df, pd.DataFrame(dt, columns = ['dt'])).droplevel(1)
            mode = 'w' if i == 0 else 'a' 
            header = i == 0
            dfx.columns = ['l','d','di', 'nh', 'a', 'p','n', 'cstar', 'dthroat']
            dfx[1:].to_csv("dataset.csv", 
                    mode= mode,
                    header = header, 
                    index = False)
        #CHUNK ENDS
        
        
        #HERE, WE ARE ADDING LISTS TO THE CSV
        df_iterator = pd.read_csv('dataset.csv', 
                            dtype={'l' : float, 'd':float, 'di':float, 'nh':float, 'a':float, 'p':float, 'n':float, 'cstar':float, 'dthroat':float}, 
                            chunksize=1e6, 
                            # header = None, 
                            on_bad_lines='skip')    

        for i, df in enumerate(df_iterator):
            
            df['InitVolume'] = pi/4 * (df.d**2 -df.nh*df.di**2) * df.l
            df['InitSA'] = math.pi * df.d**2 /4 * df.l + math.pi * df.di**2 /4 * df.l * df.nh
                     
            # df['Volumes'] = ''
            # df['SurfaceAreas'] = ''
            # df['AreaBurnt'] = ''
            df['Pressure'] = ''
            df['BurnRate'] = ''
            
            # print("chktp1", sys.stdout)
            for index, row in df.iterrows():
                # df.at[index, 'Volumes'] = getVolume(row.l, row.d, row.di, row.nh, row.a, row.p, row.n, row.br, VolStep)
                sa_list, ab_list, p_list = getSA_AB_P(row.l, row.d, row.di, row.nh, row.a, row.p, row.n, row.dthroat, row.cstar, SAStep)
                # df.at[index, 'SurfaceAreas'] = sa_list
                # df.at[index, 'AreaBurnt'] = ab_list
                # df.at[index, 'Pressure'] = p_list
                df.at[index, "BurnRate"] = [row.a * pTemp**row.n for pTemp in p_list]
            # print("chktp2", sys.stdout)
            mode = 'w' if i == 0 else 'a'
            header = i == 0
                       
            df.to_csv(
                "dataset.csv",
                index=False,  # Skip index column
                header=header, 
                mode=mode)

            # df.to_csv(
            #     "dataset.csv.gz",
            #     index=False,  # Skip index column
            #     header=header, 
            #     mode=mode,
            #     compression='gzip')
            print("Program is on the {} iteration".format(i), file = sys.stdout)

        return render_template("databaseReport.html")




# @app.route('/showVar/<varName>')
# def showVar(varName):
#     return 'Var name is: ' + str(varName)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
