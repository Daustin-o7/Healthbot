from tkinter.filedialog import Open
from unicodedata import name


def write() :
    
    outfile = open('User_data.txt','w')
    outfile.write("**********USER MEDICAL DETAILS**********\n")
    
    outfile.write("Full name :")
    fname = input("Please Enter your Full name:")
    outfile.write(fname)

    outfile.write("\nAddress:")
    Address = input("Please Enter your Address:")
    outfile.write(Address)

    outfile.write("\nAge :")
    Age = input("Please Enter your Age:")
    outfile.write(Age)

    outfile.write("\nGender:")
    Gender = input("Please Enter your Gender(M/F/Prefer not to say):")
    outfile.write(Gender)

    outfile.write("\nBody Temperature:")
    Temperature = input("What is your Body Temperature(in Celcius):")
    outfile.write(Temperature)
    degree_sign = u"\N{DEGREE SIGN}"
    outfile.write(degree_sign)
    outfile.write("C")

    outfile.write("\nBlood Pressure:")
    B_Presureup = input("What is your Blood Pressure(Upper number):")
    B_Presuredown = input("What is your Blood Pressure(lower number):")
    outfile.write(B_Presureup)
    outfile.write("/")
    outfile.write(B_Presuredown)

    outfile.write("\nNo.of Medicines per day:")
    Medicine_number = input ("Please enter the number of medicine you take per day.")
    outfile.write (Medicine_number)
    
    if(int(Medicine_number) == 0 ):
        outfile.close()
    else :
        outfile.write("\nNames of medicine and their Doses:\n")
        i = 1
        while i <= int(Medicine_number) :
            Medicine_name = input ("Please enter the name of "+str(i) +"st medicine:")
            Dose = input ("Please enter the medicine Dose:")
            outfile.write (str(i))
            outfile.write (".")
            outfile.write (Medicine_name)
            outfile.write ("\t")
            outfile.write (Dose)
            outfile.write("\n")
            i += 1

    outfile = open('User_data.txt','a+')
    outfile.write("\n\nShort report according to primary data:")
    if(36.1<=float(Temperature)<=37.2):
        outfile.write("\n1.Your body temperature seems to be normal temperature.")
    if(float(Temperature)>37.2):
        outfile.write("\n1.Your body temperature seems to be High, You might be suffering from fever.Please Contact a Doctor as soon as possible.")
    if(float(Temperature)<36.1):
      outfile.write("\n1.Your body temperature seems to be low, You might be suffering from Hypothermia.Please Contact a Doctor as soon as possible.")  
    
    
    if(180<=int(B_Presureup) and 120<=int(B_Presuredown)):
        outfile.write("\n2.The Blood Pressure Seems to be Very High, Might be the case of Stage 3 Hypertension, Contact the Doctor straight away.")
    if(140<=int(B_Presureup) and 90<=int(B_Presuredown)):
        outfile.write("\n2.Your Blood Pressure Seems to be High, Might be the case of Stage 2 Hypertension, Contact the Doctor immediately.")
    if(130<=int(B_Presureup)<=139 and 80<=int(B_Presuredown)<=89):
        outfile.write("\n2.Your Blood Pressure Seems to be High, Might be the case of Stage 1 Hypertension, Contact the Doctor.")
    if(120<int(B_Presureup)<=129 and int(B_Presuredown)<80):
        outfile.write("\n2.Your Blood Pressure Seems to be elevated.")
    if(90<=int(B_Presureup)<=120 and 60<=int(B_Presuredown)<=80):
        outfile.write("\n2.Your Blood Pressure Seems to be normal.")
    if(90>int(B_Presureup) and int(B_Presuredown)<60):
        outfile.write("\n2.Your Blood Pressure Seems to be low ,Go ahead and take a chocolate..\n")
    outfile.close()

def read ():
    Infile = open("User_data.txt",'r')
    print(Infile.read())


#write()
#read()