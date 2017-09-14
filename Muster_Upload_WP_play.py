# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:01:26 2017

@author: z002ffcz
"""

import os
import pandas as pd
import sqlite3 as lite
import Summary_Py_Functions
import sys


default_path_Py = "C:\AlexV\Python_stuff"

# Set my working directory
#--------------------------
os.chdir(default_path_Py)

#print(os.getcwd()); # Prints the working directory


# Open file connection (closing not required with 'with')
#----------------------------------------------------------
with open("Wertpapier_test.txt",mode = "r") as f:  
  #data = f.read() ; 
  data = pd.read_table(f, header=0, sep="\t")
  #data2 = f.readlines()  
#----------------------------------------------------------
print(data) ;       # show data


#-----------------------------------------------------------------------------------
# We have now created a database with the name "company". 
# It's like having sent the command "CREATE DATABASE company;" to a SQL server.
#-----------------------------------------------------------------------------------
connection = sqlite3.connect("company.db")

# To be capable to send a command to "SQL", or SQLite, we need a cursor object:
cursor = connection.cursor()     # get the cursor object by calling the cursor() method of connection

# delete 
#cursor.execute("""DROP TABLE employee;""")

sql_command = """
CREATE TABLE employee ( 
staff_number INTEGER PRIMARY KEY, 
fname VARCHAR(20), 
lname VARCHAR(30), 
gender CHAR(1), 
joining DATE,
birth_date DATE);"""

cursor.execute(sql_command)

sql_command = """INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
    VALUES (NULL, "William", "Shakespeare", "m", "1961-10-25");"""
cursor.execute(sql_command)


sql_command = """INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
    VALUES (NULL, "Frank", "Schiller", "m", "1955-08-17");"""
cursor.execute(sql_command)

cursor.execute("SELECT * FROM employee")

# We have a list with data of persons which will be used in the INSERT statement:
staff_data = [ ("William", "Shakespeare", "m", "1961-10-25"),
               ("Frank", "Schiller", "m", "1955-08-17"),
               ("Jane", "Wall", "f", "1989-03-14") ]
               
for p in staff_data:
    format_str = """INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
    VALUES (NULL, "{first}", "{last}", "{gender}", "{birthdate}");"""

    sql_command = format_str.format(first=p[0], last=p[1], gender=p[2], birthdate = p[3])
    cursor.execute(sql_command)


# never forget this, if you want the changes to be saved:
connection.commit()

connection.close()
#-------------------------------------------------------------------------------

data.head()

nrow = data.shape[0]
nrow
ncol = data.shape[1]
ncol

# Faster
for row in data.itertuples(index=True):
    print(getattr(row, "AssetType"), getattr(row, "EODPrice"))


for i in range(1,nrow):
    print (data.iloc[i]['AssetType'], data.iloc[i]['EODPrice'])

data.iloc[1,4]


# The following block of code establishes a database, sets up a table 
# which can be checked in SQLite Studio then

#---------------------------------------------------------
# Establish and connecting to a database connection:
#---------------------------------------------------------    
connection = lite.connect("mytest.db")

cursor = connection.cursor()     # get the cursor object by calling the cursor() method of connection

cursor.execute("""DROP TABLE Wertpapiere;""")

sql_command = """
CREATE TABLE Wertpapiere ( 
AssetType VARCHAR(30), 
Name VARCHAR(30), 
Date DATE,
Identifier VARCHAR(30),
EODPrice FLOAT);"""
cursor.execute(sql_command)


sql_command = """INSERT INTO Wertpapiere (AssetType, Name, Date, Identifier	, EODPrice)
    VALUES ("QOT","SKAG_WP_SPE", "2017-07-27", "DE000A0KEXM6", 17.925);"""
cursor.execute(sql_command)

#cursor.execute("SELECT * FROM Wertpapiere")

cursor.execute("""DROP TABLE myTable;""")

sql_command = """
CREATE TABLE myTable ( 
AssetType VARCHAR(30), 
Name VARCHAR(30), 
Date DATE,
Identifier VARCHAR(30),
EODPrice REAL);"""
cursor.execute(sql_command)

i=0;
for row in data.itertuples(index=True):
    print(i)
    #print(getattr(row, "AssetType"), getattr(row, "Identifier"), getattr(row, "EODPrice"))
    cursor.execute("""INSERT INTO myTable (AssetType, Name, Date, Identifier, EODPrice) 
    VALUES(?, ?, ?, ?, ?)""", (row[1], row[2], row[3], row[4], row[5]))
    i += 1; 

connection.commit()

connection.close()
#------------------------------------------------------------------------------

con = None           # set connection 

with con:
    
    cur = con.cursor()    
    cur.execute("CREATE TABLE Cars(Id INT, Name TEXT, Price INT)")
    cur.execute("INSERT INTO Cars VALUES(1,'Audi',52642)")
    cur.execute("INSERT INTO Cars VALUES(2,'Mercedes',57127)")
    cur.execute("INSERT INTO Cars VALUES(3,'Skoda',9000)")
    cur.execute("INSERT INTO Cars VALUES(4,'Volvo',29000)")
    cur.execute("INSERT INTO Cars VALUES(5,'Bentley',350000)")
    cur.execute("INSERT INTO Cars VALUES(6,'Citroen',21000)")
    cur.execute("INSERT INTO Cars VALUES(7,'Hummer',41400)")
    cur.execute("INSERT INTO Cars VALUES(8,'Volkswagen',21600)")
#------------------------------------------------------------------------        
con.close()



se = sequence(1,nrow)
    
data = data.assign(Index=se)

data.head()
print(data.index)

data.DataFrame(Index=se:data)

























