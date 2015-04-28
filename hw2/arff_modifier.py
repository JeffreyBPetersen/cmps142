#!/usr/bin/env python3
# arff_modifier.py
# So far only adds 10 copies of an attribute to the diabetes_modified.arff file

import linecache
from random import randint
def main():
   data_file = open('diabetes_data', 'w')
   
   """ for i in range(107, 875):
      print(i)
      line = linecache.getline('diabetes_modified.arff', i)
      print(line)
      line_list = line.split(',')
      print(line_list)
      age = line_list[7]
      print('Age = ', age)
      for j in range(10):
         line_list.insert(7, age)
      print(line_list)
      newline = ", ".join(line_list)
      print(newline)
      data_file.write(newline) """
      
      
   for i in range(117, 885):
      print(i)
      line = linecache.getline('diabetes_modified2.arff', i)
      print(line)
      line_list = line.split(',')
      print(line_list)
      for j in range(20):
         value = str(randint(0,1))
         line_list.insert(8, value)
      print(line_list)
      newline = ", ".join(line_list)
      print(newline)
      data_file.write(newline)

   data_file.close()
   

if __name__ == '__main__':
   main()
