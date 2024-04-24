#!/usr/bin/env python
'''
This code initializes the rfsoc4x2 board. Code copied from adc16g_init:
-- Refer to Mitch Burnett's website for details.
-- Homin 2021-0924 init. 
-- 

 This plot is for ADC and spectrum combined.
 This code is actually designed for zcu 216, b DR Homin an di modified it to 4x2 board
'''
 # This design code is form DR homin design of simple spectrometer

#TODO: 
#TODO:  

import casperfpga,time,numpy,struct,sys,logging,pylab,matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import casperfpga
import numpy as np
from rfsoc4x2_tut_rfdc_utils_200mhz import *

#katcp_port=7147

snap_len = 512
snap_adc_len = 1024 # The length of ADC is set to 256.
#freq_range_mhz = numpy.linspace(0., 400., 2048)

#snap_len = 100
snap_adc_name = 'adc' #The name of ADC used
snap_name = 'adc'
snap_adc_name2 = 'adc1'
snap_name2 = 'adc1'
snap_adc_name3 = 'adc2'
snap_name3 = 'adc2'
snap_adc_name4 = 'adc3'
snap_name4 = 'adc3'
snap_adc_name5 = 'adc4'
snap_name5 = 'adc4'
# Get the data from ADC#
def snap_data():
    

 #   a_0=struct.unpack('>1024Q',fpga.read('even',1024*8,0))
 #   a_1=struct.unpack('>1024Q',fpga.read('odd',1024*8,0))

#    a_0=struct.unpack('>1024I',fpga.read('snapshot_bram',1024*4,0))

    fpga.write_int(snap_adc_name+'_ctrl',0,0) # This is the one which give pulse to it.
    fpga.write_int(snap_adc_name+'_ctrl',1,0)
    fpga.write_int(snap_adc_name+'_ctrl',0,0)

#    a_0=struct.unpack('>%iI'%snap_len,fpga.read(snap_name+'_bram',snap_len*4,0))
    a_0=struct.unpack('>%iI'%snap_adc_len,fpga.read(snap_adc_name+'_bram',snap_adc_len*4,0)) #This line of the script snap the data for ADC
    X = a_0[100:]
    #Xa = a_0[-100:]
    print (X)
    #print (Xa)
    print ('The number of data ADC processed: ',len (a_0))
    interleave_a=[]

#    for i in range(1024):
    for i in range(snap_adc_len):

        interleave_a.append(a_0[i])
#        interleave_a.append(a_1[i])
#    return acc_n, numpy.array(interleave_a,dtype=numpy.float) 
    return numpy.array(interleave_a,dtype=numpy.int64) 
    
#----------------------------
def plot_adc():

 
    interleave_a = snap_data() # The data from RFDC to adc_snapshot

    print ('trying interleave ... \n')

    print (type(interleave_a))

    
   
    pylab.figure(num=1,figsize=(11,11))
    pylab.clf()
    pylab.subplot(111)
    pylab.title('ADC Clock at 1600 Msps ')
    pylab.plot(interleave_a[:100])
   
    pylab.ylabel('Count') 
    pylab.grid()
    pylab.xlabel('sample')
    
    pylab.xlim(0,len(interleave_a[100:])) 
    pylab.show()
    
    # fname ='./adc10g_all%d'%cnt_plot
################### The plot for adc1, newly added adc
def snap_data2():
    

 #   a_0=struct.unpack('>1024Q',fpga.read('even',1024*8,0))
 #   a_1=struct.unpack('>1024Q',fpga.read('odd',1024*8,0))

#    a_0=struct.unpack('>1024I',fpga.read('snapshot_bram',1024*4,0))

    fpga.write_int(snap_adc_name2+'_ctrl',0,0) # This is the one which give pulse to it.
    fpga.write_int(snap_adc_name2+'_ctrl',1,0)
    fpga.write_int(snap_adc_name2+'_ctrl',0,0)

#    a_0=struct.unpack('>%iI'%snap_len,fpga.read(snap_name+'_bram',snap_len*4,0))
    a_0=struct.unpack('>%iI'%snap_adc_len,fpga.read(snap_adc_name2+'_bram',snap_adc_len*4,0)) #This line of the script snap the data for ADC
    #X2 = a_0[-100:]  # For last 100 data
    X2b = a_0[:100]
    #print (X2)
    print (X2b)
    print ('The number of data ADC2 processed: ',len (a_0))
    interleave_a=[]

#    for i in range(1024):
    for i in range(snap_adc_len):

        interleave_a.append(a_0[i])
#        interleave_a.append(a_1[i])
#    return acc_n, numpy.array(interleave_a,dtype=numpy.float) 
    return numpy.array(interleave_a,dtype=numpy.int64) 
    
#----------------------------
def plot_adc2():

 
    interleave_a = snap_data2() # The data from RFDC to adc_snapshot

    print ('trying interleave ... \n')

    print (type(interleave_a))

    
   
    pylab.figure(num=1,figsize=(11,11))
    pylab.clf()
    pylab.subplot(111)
    pylab.title('ADC Clock at 1600 Msps ')
    pylab.plot(interleave_a[:100])
   
    pylab.ylabel('Count (8bit)') 
    pylab.grid()
    pylab.xlabel('sample')
    
    pylab.xlim(0,len(interleave_a[:100])) 
    pylab.show()
########### 3rd ADC
def snap_data3():
    

 #   a_0=struct.unpack('>1024Q',fpga.read('even',1024*8,0))
 #   a_1=struct.unpack('>1024Q',fpga.read('odd',1024*8,0))

#    a_0=struct.unpack('>1024I',fpga.read('snapshot_bram',1024*4,0))

    fpga.write_int(snap_adc_name3+'_ctrl',0,0) # This is the one which give pulse to it.
    fpga.write_int(snap_adc_name3+'_ctrl',1,0)
    fpga.write_int(snap_adc_name3+'_ctrl',0,0)

#    a_0=struct.unpack('>%iI'%snap_len,fpga.read(snap_name+'_bram',snap_len*4,0))
    a_0=struct.unpack('>%iI'%snap_adc_len,fpga.read(snap_adc_name3+'_bram',snap_adc_len*4,0)) #This line of the script snap the data for ADC
    #X2 = a_0[-100:]  # For last 100 data
    X2b = a_0[:100]
    #print (X2)
    print (X2b)
    print ('The number of data ADC3 processed: ',len (a_0))
    interleave_a=[]

#    for i in range(1024):
    for i in range(snap_adc_len):

        interleave_a.append(a_0[i])
#        interleave_a.append(a_1[i])
#    return acc_n, numpy.array(interleave_a,dtype=numpy.float) 
    return numpy.array(interleave_a,dtype=numpy.int64) 
    
#----------------------------
def plot_adc3():

 
    interleave_a = snap_data3() # The data from RFDC to adc_snapshot

    print ('trying interleave ... \n')

    print (type(interleave_a))

    
   
    pylab.figure(num=1,figsize=(11,11))
    pylab.clf()
    pylab.subplot(111)
    pylab.title('ADC Clock at 1600 Msps ')
    pylab.plot(interleave_a[:100])
   
    pylab.ylabel('Count (8bit)') 
    pylab.grid()
    pylab.xlabel('sample')
    
    pylab.xlim(0,len(interleave_a[:100])) 
    pylab.show()
################### ADC 4
def snap_data4():
    

 #   a_0=struct.unpack('>1024Q',fpga.read('even',1024*8,0))
 #   a_1=struct.unpack('>1024Q',fpga.read('odd',1024*8,0))

#    a_0=struct.unpack('>1024I',fpga.read('snapshot_bram',1024*4,0))

    fpga.write_int(snap_adc_name4+'_ctrl',0,0) # This is the one which give pulse to it.
    fpga.write_int(snap_adc_name4+'_ctrl',1,0)
    fpga.write_int(snap_adc_name4+'_ctrl',0,0)

#    a_0=struct.unpack('>%iI'%snap_len,fpga.read(snap_name+'_bram',snap_len*4,0))
    a_0=struct.unpack('>%iI'%snap_adc_len,fpga.read(snap_adc_name4+'_bram',snap_adc_len*4,0)) #This line of the script snap the data for ADC
    #X2 = a_0[-100:]  # For last 100 data
    X2b = a_0[:100]
    #print (X2)
    print (X2b)
    print ('The number of data ADC4 processed: ',len (a_0))
    interleave_a=[]

#    for i in range(1024):
    for i in range(snap_adc_len):

        interleave_a.append(a_0[i])
#        interleave_a.append(a_1[i])
#    return acc_n, numpy.array(interleave_a,dtype=numpy.float) 
    return numpy.array(interleave_a,dtype=numpy.int64) 
    
#----------------------------
def plot_adc4():

 
    interleave_a = snap_data4() # The data from RFDC to adc_snapshot

    print ('trying interleave ... \n')

    print (type(interleave_a))

    
   
    pylab.figure(num=1,figsize=(11,11))
    pylab.clf()
    pylab.subplot(111)
    pylab.title('ADC Clock at 1600 Msps ')
    pylab.plot(interleave_a[:100])
   
    pylab.ylabel('Count (8bit)') 
    pylab.grid()
    pylab.xlabel('sample')
    
    pylab.xlim(0,len(interleave_a[:100])) 
    pylab.show()
################ ADC5
def snap_data5():
    

 #   a_0=struct.unpack('>1024Q',fpga.read('even',1024*8,0))
 #   a_1=struct.unpack('>1024Q',fpga.read('odd',1024*8,0))

#    a_0=struct.unpack('>1024I',fpga.read('snapshot_bram',1024*4,0))

    fpga.write_int(snap_adc_name5+'_ctrl',0,0) # This is the one which give pulse to it.
    fpga.write_int(snap_adc_name5+'_ctrl',1,0)
    fpga.write_int(snap_adc_name5+'_ctrl',0,0)

#    a_0=struct.unpack('>%iI'%snap_len,fpga.read(snap_name+'_bram',snap_len*4,0))
    a_0=struct.unpack('>%iI'%snap_adc_len,fpga.read(snap_adc_name5+'_bram',snap_adc_len*4,0)) #This line of the script snap the data for ADC
    #X2 = a_0[-100:]  # For last 100 data
    X2b = a_0[:100]
    #print (X2)
    print (X2b)
    print ('The number of data ADC4 processed: ',len (a_0))
    interleave_a=[]

#    for i in range(1024):
    for i in range(snap_adc_len):

        interleave_a.append(a_0[i])
#        interleave_a.append(a_1[i])
#    return acc_n, numpy.array(interleave_a,dtype=numpy.float) 
    return numpy.array(interleave_a,dtype=numpy.int64) 
    
#----------------------------
def plot_adc5():

 
    interleave_a = snap_data5() # The data from RFDC to adc_snapshot

    print ('trying interleave ... \n')

    print (type(interleave_a))

    
   
    pylab.figure(num=1,figsize=(11,11))
    pylab.clf()
    pylab.subplot(111)
    pylab.title('ADC Clock at 1600 Msps ')
    pylab.plot(interleave_a[:100])
   
    pylab.ylabel('Count (8bit)') 
    pylab.grid()
    pylab.xlabel('sample')
    
    pylab.xlim(0,len(interleave_a[:100])) 
    pylab.show()
#############

'''def get_vacc_data(fpga, nchannels=4, nfft=2048):
  acc_n = fpga.read_uint('acc_cnt')
  chunk = nfft//nchannels
  print('nchannels = ', nchannels)
  print('nfft = ', nfft)
  print('acc_n = ', acc_n)
  print('chunk = ', chunk)
  raw = np.zeros((nchannels, chunk))
  for i in range(nchannels):
    raw[i,:] = struct.unpack('>{:d}Q'.format(chunk), fpga.read('q{:d}'.format((i+1)),chunk*8,0))
    print('raw = ', raw, '\n')	
  interleave_q = []
  for i in range(chunk):
    for j in range(nchannels):
      interleave_q.append(raw[j,i])
  num_channels = len(raw)  
  num_samples_per_channel = len(raw[0]) 
  print('type of raw[0][0] = ', type(raw[0][0]))
  print("Number of channels:", num_channels)
  print("Number of samples per channel:", num_samples_per_channel)
  print('type(raw) = ', type(interleave_q))
  print('\n\n')
  return acc_n, np.array(interleave_q, dtype=np.float64)'''

def get_vacc_data(fpga, nchannels=4, nfft=4096): # Change the channels  of FFT outputs
  acc_n = fpga.read_uint('acc_cnt')
  print ('Accumulation count = ', acc_n)
  chunk = nfft//nchannels
  print ("Number of chunks", chunk)
  raw = np.zeros((nchannels, chunk))
  print ('Number of Raw data:',len(raw))
  print (type(raw))
  for i in range(nchannels):
    raw[i,:] = struct.unpack('>{:d}Q'.format(chunk), fpga.read('q{:d}'.format((i+1)),chunk*8,0))
    #raw[i, :] = np.frombuffer(fpga.read('q{:d}'.format((i + 1)), chunk * 8, 0), dtype=np.uint64)
    print ('The data from ADC', raw[i,:]) # The data from ADC
    print (type(raw[i,:]))
  interleave_q = []
  for i in range(chunk):
    for j in range(nchannels):
      interleave_q.append(raw[j,i])
      #print ('Interleaved_data =', interleave_q)
  return acc_n, np.array(interleave_q, dtype=np.float64)
#represents the number of samples per channel that will be read from the FPGA in each iteration of the loop
def plot_spectrum(fpga, cx=True, num_acc_updates=None):
  

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.grid();

  fs = 1600/1
  if cx:
    print('complex design')
    Nfft = 2**11
    fbins = np.arange(-Nfft//2, Nfft//2)
    nchannels = 8 
    df = fs/Nfft
    faxis = fbins*df + fs/2
  else:
    print('real design')
    Nfft = 2**12   # This is the number of bins of data that is flowing from the fft.
    fbins = np.arange(0, Nfft//2)
    df = fs/Nfft
    nchannels = 4 # After fft how many outputs are going to be received
    faxis = fbins*df

  if cx:
    acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft)
    line, = ax.plot(faxis,10*np.log10(fft.fftshift(spectrum)),'-')
  else:
    acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft//2)
    line, = ax.plot(faxis,10*np.log10(spectrum),'-')

  ax.set_xlabel('Frequency (MHz)')
  ax.set_ylabel('Power (dB arb.)')

  def update(frame, *fargs):
    if cx:
      acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft)
      line.set_ydata(10*np.log10(fft.fftshift(spectrum)))
    else:
      acc_n, spectrum = get_vacc_data(fpga, nchannels=nchannels, nfft=Nfft//2)
      line.set_ydata(10*np.log10(spectrum))

    ax.set_title('acc num: %d' % acc_n)

  v = anim.FuncAnimation(fig, update, frames=1, repeat=True, fargs=None, interval=500)
  #anim.FuncAnimation(fig, update, frames=1, repeat=True, fargs=None, interval=1000)
  #v.save('animation.mp4', writer='ffmpeg')
  plt.show()
############################3
if __name__=="__main__":
  from optparse import OptionParser

  p = OptionParser()
  p.set_usage('frb_fpga_modified_part2_2.py <HOSTNAME_or_IP> cx|real [options]')
  p.set_description(__doc__)
  p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=2*(2**28)//2048,
      help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048')
  p.add_option('-s', '--skip', dest='skip', action='store_true',
      help='Skip programming and begin to plot data')
  p.add_option('-b', '--fpg', dest='fpgfile',type='str', default='',
      help='Specify the fpg file to load')
  p.add_option('-c', '--channel_sel', dest='adc_chan_sel', type='int',default=0,
      help='Select 0-3 ADC channel to monitor. default is channel 0.')

  opts, args = p.parse_args(sys.argv[1:])
  if len(args) < 2:
    print('Specify a hostname or IP for your casper platform. And either cx|real to indicate the type of spectrometer design.\n'
          'Run with the -h flag to see all options.')
    exit()
  else:
    hostname = args[0]
    mode_str = args[1]
    print("mode=", mode_str)
    if mode_str=='cx':
      mode = 1
    elif mode_str=='real':
      mode = 0
    else:
      print('operation mode not recognized, must be "cx" or "real"')
      exit()

  if opts.fpgfile != '':
    bitstream = opts.fpgfile
  else:
    if mode == 1:
      fpg_prebuilt = '/home/bishnu/Desktop/m_work/tutorials_devel/rfsoc/tut_rfdc/prebuilt/rfsoc4x2/rfsoc4x2_tut_rfdc_cx.fpg'
    else:
      fpg_prebuilt = '/home/bishnu/Desktop/m_work/tutorials_devel/rfsoc/tut_rfdc/prebuilt/rfsoc4x2/rfsoc4x2_tut_rfdc_real.fpg'

    print('using prebuilt fpg file at %s' % fpg_prebuilt)
    bitstream = fpg_prebuilt

  print('Connecting to %s... ' % (hostname))
  fpga = casperfpga.CasperFpga(hostname)
  time.sleep(0.2)

  if not opts.skip:
    print('Programming FPGA with %s...'% bitstream)
    fpga.upload_to_ram_and_program(bitstream)
    print('done')
  else:
    fpga.get_system_information()
    print('skip programming fpga...')
  
  fpga.listdev ()
  print("list devices = %s \n"%fpga.listdev())
    
  clk=fpga.estimate_fpga_clock()	
  print('Configuring accumulation period...')
  fpga.write_int('acc_len',opts.acc_len)
  time.sleep(0.1)
  print('done')
  print ("\n\n\nFPGA clock = %s MHz\n\n\n"%clk)
  time.sleep(1.0)

  print ("clock counter = %s \n"%(fpga.read_int('sys_clkcounter')))
  time.sleep(1.0)
  rfdc_rfsoc4x2 = fpga.adcs['rfdc']
  print ('RFDC = ', rfdc_rfsoc4x2)
  time.sleep(3.0)
   
  if ( rfdc_rfsoc4x2.init() == True ):
      print ("ADC init. OK \n")
      print("ADC status = %s \n"%rfdc_rfsoc4x2.status())

  else :
      print ("ADC init. failed \n")
      exit_fail()
  print ('Specify the ADC channel ...')
  c=rfdc_rfsoc4x2.show_clk_files()
  print ('The clock file that are inside  FPGA: cd /lib/firmware are :', c)
  
##################################        
  print('Resetting counters...')
  fpga.write_int('cnt_rst',1) 
  fpga.write_int('cnt_rst',0) 
  time.sleep(5)
  print('done')
  clk=fpga.estimate_fpga_clock()	

  print ("\n\n\nFPGA clock  is = %s MHz\n\n\n"%clk)            
    
  time.sleep(0.2)
  rfdc_rfsoc4x2.status()
  
  
  
  try:
    plot_adc()
    plot_adc2()
    plot_adc3()
    plot_adc4()
    plot_adc5()
    plot_spectrum(fpga, cx=mode)
  except KeyboardInterrupt:
    exit()
  
 
