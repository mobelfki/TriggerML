#!/usr/bin/env python
from DataProcessing.DataProcessing import SuperCells
from Models.models import *
from Helps.helps import *
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import json
import os
seed = 412
import scipy.stats as stats
import numpy as np
np.random.seed(seed)
from numpy.random import seed as random_seed
random_seed(seed)
import matplotlib.pyplot as plt  
import math

def getArgs():
	
	args = ArgumentParser(description="Argumetns")
	args.add_argument('-d', '--dir', action='store', help='json config file')
	args.add_argument('-c', '--plot_config', action='store', help='json config file for plots')
	args.add_argument('-t', '--doTrig', action='store', default=False, help='do Trigger plots')
	return args
	
def getSamples(args):

	dic = {410471: 'ttbar', 345058: 'ggZH(vv)', 361108: 'Ztautau', 361020: 'JZ0W', 0:'TEST', 1: 'TEST2', 999999: 'Combined', 361021: 'JZ1W'}
	return dic[args]	
	
def getStrategy(args):
		
		if int(args.strategy) == 2:
			return 'ratio_to_%s'%(args.learn_ratio_to)
		if int(args.strategy) == 1:
			return 'Truth MET'
		return 'None'

def getTag(tag):
	dic = {"CellsSum": "Sum of Cells", "Number": "Event Number", "Weight": "Weight", "LiveTime": "Event live time", "Density": "Event density", "DensitySigma": "Event density Sig", "Area": "Event area", "isGoodLB": "Good LB", "distFrontBunchTrain": "DFBT", "mu": "pile-up", "L1_XE30": "L1XE30", "L1_XE300": "L1XE300", "L1_XE35": "L1XE35", "L1_XE40": "L1XE40", "L1_XE45": "L1XE45", "L1_XE50": "L1XE50", "L1_XE55": "L1XE55", "L1_XE60": "L1XE60", "cell_ex": "HLT Cell eX", "cell_ey": "HLT Cell eY", "cell_et": "HLT Cell eT", "mht_ex": "HLT MHT eX", "mht_ey": "HLT MHT eY", "mht_et": "HLT MHT eT" , "topocl_PUC_ex": "HLT Topocl eX (PUC)", "topocl_PUC_ey": "HLT Topocl eY (PUC)", "topocl_PUC_et": "HLT Topocl eT (PUC)", "cell_PUC_ex": "HLT Cell eX (PUC)", "cell_PUC_ey": "HLT Cell eY (PUC)", "cell_PUC_et": "HLT Cell eT (PUC)", "Truth_MET_NonInt_ex": "Truth MET eX", "Truth_MET_NonInt_ey": "Truth MET eY", "Truth_MET_NonInt_et": "Truth MET eT", "NN_et": "CNN eT", "predicted": "Predicted", "truth": "Target", "scell_ex": "HLT SCell eX", "scell_ey": "HLT SCell eY", "scell_et": "HLT SCell eT", "Truth_MET_NonMuons_ex": "Truth MET eX [NoMuon]", "Truth_MET_NonMuons_ey": "Truth MET eY [NoMuon]", "Truth_MET_NonMuons_et": "Truth MET eT [NoMuon]", "predicted_x": "Predicted Ex", "predicted_y": "Predicted Ey", "predicted_met": "Predicted MET", "truth_x": "Truth Ex", "truth_y": "Truth Ey", "truth_met": "Truth MET"}			

	try: 
		return dic[tag]
	except:
		return tag
def getMETAlgo(tag):

	dic = {"scellAlgo": "scell_et", "cellAlgo": "cell_et", "NNAlgo": "predicted_met", "mhtAlgo": "mht_et", "topoAlgo": "topocl_PUC_et"}
	
	return dic[tag]

def getAlgoName(tag):

	dic = {"scellAlgo": "HLT SCell", "cellAlgo": "HLT Cell", "NNAlgo": "CNN", "mhtAlgo": "HLT MHT", "topoAlgo": "HLT Topocl PUC"}
	
	return dic[tag]
	
		
def plot_loss(args, data):
	
	plt.figure()
	plt.plot(data.epoch+1, data.train_loss) 
	plt.plot(data.epoch+1, data.val_loss)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Training loss', 'Validation loss'], )
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, loss: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), args.loss))
	plt.savefig('%s/plots/loss.pdf'%(args.output_dir))
	plt.close()
	
def load_df(args):

	path = args.output_dir+'/results/'
	files  = os.listdir(path)
	print('--following result found :',files)
	dic = {}
	for file in files: 
		DSID = file.split('.')[1]
		dic.update({int(DSID): pd.read_hdf(path+file)})
	
	return dic
	
def histVar(args, data, dsid, var, cut):
	
	plt.figure()
	values = data
	if vars(args)[var]['unit'] == 'MeV': values = values / 1000
	plt.hist(values, bins=vars(args)[var]['bins'], range=vars(args)[var]['range'], density=vars(args)[var]['density'], histtype=vars(args)[var]['type'])
	plt.xlabel(vars(args)[var]['xlabel'])
	plt.ylabel(vars(args)[var]['ylabel'])
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, test on: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), getSamples(dsid)))
	legend = plt.legend([getTag(var)])
	legend.set_title("%s, L1: %s"%(getSamples(dsid), getTag(cut)))
	plt.savefig('%s/plots/var_%s_smaple_%s_model_%s_Cut_%s.pdf'%(args.output_dir, var, getSamples(dsid), args.model_name, cut))
	plt.close()

def hist(args, data, var):
	
	for key, item in data.items():	
		L1_triggers = [col for col in item.columns if 'L1' in col]
		histVar(args, item[var], key, var, 'NONE')
	
		if args.doTrig:	
			for L1 in L1_triggers:
				mask = item[L1] == 1
				histVar(args, item[var][mask], key, var, L1)
			

def profile(args, data, var):

	for key, item in data.items():
	
		L1_triggers = [col for col in item.columns if 'L1' in col]
		mask = np.ones(item.shape[0],) == 1
		profileVars(args, item, key, var, mask, 'NONE')
		if args.doTrig:
			for L1 in L1_triggers: 
				mask = item[L1] == 1
				profileVars(args, item, key, var, mask, L1)

def profileVars(args, data, dsid, var, mask, cut):

	plt.figure()
	
	hists = vars(args)[var]['vars']
	
	for hist in hists:
		values = data[hist][mask]		
		ref    = data[vars(args)[var]['ref']][mask]
		if vars(args)[var]['Varunit'] == 'MeV': values = values / 1000
		if vars(args)[var]['Refunit'] == 'MeV': ref = ref / 1000
		
		tmp_df = pd.concat([values, ref], axis = 1)	
		bincontent, bins=   getBinContents(ref, args, var)
		labels = range(0, len(bins)-1)
		tmp_df['bin'] = pd.cut(tmp_df[vars(args)[var]['ref']], bins=bins, labels=labels).astype("float")
		tmp_m = []
		tmp_s = []
		tmp_c = []
		tmp_df['bin'] = tmp_df['bin'].replace(np.nan, len(bins))
		
		for l in labels:
			v = tmp_df[hist][tmp_df['bin'] == l]
			tmp_c.append(len(v))
			tmp_m.append(v.mean())
			tmp_s.append(v.std())
		bin_center = 0.5*(bins[1:]+bins[:-1])
		if vars(args)[var]['mode'] == "mean": 
			plt.errorbar(bin_center, tmp_m, yerr= tmp_s, fmt='o', markersize=3)
		if vars(args)[var]['mode'] == "count": plt.errorbar(bin_center, tmp_c, fmt='o', markersize=3)
		
	plt.xlabel(vars(args)[var]['xlabel'])
	plt.ylabel(vars(args)[var]['ylabel'])
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, test on: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), getSamples(dsid)))
	legend = plt.legend([ getTag(tag) for tag in vars(args)[var]['vars']])
	legend.set_title("%s, L1: %s"%(getSamples(dsid), getTag(cut)))
	plt.savefig('%s/plots/Profile_%s_smaple_%s_model_%s_Cut_%s.pdf'%(args.output_dir, vars(args)[var]['name'], getSamples(dsid), args.model_name, cut))
	plt.close()	
		
def stackVars(args, data, dsid, stack, mask, cut):
	
	plt.figure()
	hists = vars(args)[stack]['vars']
	for hist in hists:
		values = data[hist][mask]
		if vars(args)[stack]['unit'] == 'MeV': values = values / 1000
		plt.hist(values, bins=vars(args)[stack]['bins'], range=vars(args)[stack]['range'], density=vars(args)[stack]['density'], histtype=vars(args)[stack]['type'])
	plt.xlabel(vars(args)[stack]['xlabel'])
	plt.ylabel(vars(args)[stack]['ylabel'])
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, test on: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), getSamples(dsid)))
	legend = plt.legend([ getTag(tag) for tag in vars(args)[stack]['vars']])
	legend.set_title("%s, L1: %s"%(getSamples(dsid), getTag(cut)))
	plt.savefig('%s/plots/Stack_%s_smaple_%s_model_%s_Cut_%s.pdf'%(args.output_dir, vars(args)[stack]['name'], getSamples(dsid), args.model_name, cut))
	plt.close()
	
def stacks(args, data, stack):

	for key, item in data.items():
		L1_triggers = [col for col in item.columns if 'L1' in col]
		mask = np.ones(item.shape[0],) == 1
		stackVars(args, item, key, stack, mask, 'NONE')
		if args.doTrig:
			for L1 in L1_triggers: 
				mask = item[L1] == 1
				stackVars(args, item, key, stack, mask, L1) 

def resoVars(args, data, dsid, reso, mask, cut):

	plt.figure()
	hists = vars(args)[reso]['vars']
	for hist in hists:
		values = data[hist][mask]
		ref = data[vars(args)[reso]['ref']][mask]
		values = (values - ref)/ref
		plt.hist(values, bins=vars(args)[reso]['bins'], range=vars(args)[reso]['range'], density=vars(args)[reso]['density'], histtype=vars(args)[reso]['type'])
	plt.xlabel(vars(args)[reso]['xlabel'])
	plt.ylabel(vars(args)[reso]['ylabel'])
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, test on: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), getSamples(dsid)))
	legend = plt.legend([ getTag(tag) for tag in vars(args)[reso]['vars']])
	legend.set_title("%s, L1: %s"%(getSamples(dsid), getTag(cut)))
	plt.savefig('%s/plots/Resolution_%s_smaple_%s_model_%s_Cut_%s.pdf'%(args.output_dir, vars(args)[reso]['name'], getSamples(dsid), args.model_name, cut))
	plt.close()
	
def getBinContents(data, args, eff):
	
	plt.figure()
	content, bins, _ = plt.hist(data, bins=vars(args)[eff]['bins'], range=vars(args)[eff]['range'], density=vars(args)[eff]['density'], histtype=vars(args)[eff]['type'])
	plt.close()
	return content, bins
	
def getEfficiencies(args, values, eff):

	Range = vars(args)[eff]['range']
	nbin   = vars(args)[eff]['bins']
	cuts = np.linspace(Range[0], Range[1], num=nbin, endpoint=False)
	n0 = float(values.shape[0])
	rate = []
	low  = []
	high = []
	for cut in cuts:
		ispassed = values >= cut
		npassed = float(ispassed.sum()) # this should corrected when adding weights
		f, l, h = computeSingleEff(n0, npassed)
		rate.append(f)
		low.append(l)
		high.append(h)
	
	return cuts, rate, low, high		
	
def effVars(args, data, dsid, eff, mask, cut):
	
	plt.figure()
	hists = vars(args)[eff]['vars']
	for hist in hists:
		
		truth_mask = data[vars(args)[eff]['truth_var']] > vars(args)[eff]['truth_cut']*1000	
		mask = mask & truth_mask
		ref    = data[vars(args)[eff]['ref']][mask]
		values = data[getMETAlgo(hist)][mask]
		if vars(args)[eff]['Varunit'] == 'MeV': values = values / 1000
		if vars(args)[eff]['Refunit'] == 'MeV': ref = ref / 1000
		all_value = values > 0
		passed    = values > vars(args)[eff]['cut']
		n_all, bin_all = getBinContents(ref[all_value], args, eff) 
		n_passed, bin_pass = getBinContents(ref[passed], args, eff)
		bins, effs, eff_err_l, eff_err_h = computeEff(n_all, n_passed, bin_pass)
		plt.errorbar(bins, effs, yerr= [eff_err_l, eff_err_h], fmt='o', markersize=3)
		
	plt.xlabel(vars(args)[eff]['xlabel'])
	plt.ylabel("%s (Truth MET > %s GeV)"%(vars(args)[eff]['ylabel'], vars(args)[eff]['truth_cut']))
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, test on: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), getSamples(dsid)))
	legend = plt.legend([getAlgoName(tag) for tag in vars(args)[eff]['vars']])
	legend.set_title("%s, L1: %s, %s > %s"%(getSamples(dsid) , getTag(cut), vars(args)[eff]['cutName'], str(vars(args)[eff]['cut'])))
	plt.savefig('%s/plots/Efficiency_%s_smaple_%s_model_%s_L1Cut_%s_HLTCut_%s_TruthMETCut_%s.pdf'%(args.output_dir, vars(args)[eff]['name'], getSamples(dsid), args.model_name, cut, vars(args)[eff]['cut'], vars(args)[eff]['truth_cut']))
	plt.close()

	
	
def effRateVars(args, data, dsid, eff, mask, cut):

	plt.figure()
	hists = vars(args)[eff]['vars']
	for hist in hists: 
			truth_mask = data[vars(args)[eff]['truth_var']] > vars(args)[eff]['truth_cut']*1000
			mask = mask & truth_mask
			values = data[getMETAlgo(hist)][mask]
			if vars(args)[eff]['Varunit'] == 'MeV': values = values / 1000
			cuts, rate, low, high = getEfficiencies(args, values, eff)
			plt.errorbar(cuts, rate, yerr=[low, high], fmt='o', markersize=3)
	plt.xlabel(vars(args)[eff]['xlabel'])
	plt.ylabel("%s (Truth MET > %s GeV)"%(vars(args)[eff]['ylabel'], vars(args)[eff]['truth_cut']))
	plt.title('model: %s, train on: %s, L1: %s, learn: %s, test on: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args), getSamples(dsid)))
	legend = plt.legend([getAlgoName(tag) for tag in vars(args)[eff]['vars']])
	legend.set_title("%s, L1: %s"%(getSamples(dsid) ,getTag(cut)))
	plt.savefig('%s/plots/EfficiencyRate_%s_smaple_%s_model_%s_L1Cut_%s_TruthMETCut_%s.pdf'%(args.output_dir, vars(args)[eff]['name'], getSamples(dsid), args.model_name, cut, vars(args)[eff]['truth_cut']))
	plt.close()

def rocVars(args, df_sig, df_bkg, sig, bkg, roc, mask_sig, mask_bkg, cut):
	
	plt.figure()
	hists = vars(args)[roc]['vars']
	for hist in hists:
		truth_mask_sig = df_sig[vars(args)[roc]['truth_var']] > vars(args)[roc]['sig_truth_cut']*1000
		truth_mask_bkg = df_bkg[vars(args)[roc]['truth_var']] < vars(args)[roc]['bkg_truth_cut']*1000
		
		nonzero_mask_sig = df_sig[vars(args)[roc]['truth_var']] > 0
		nonzero_mask_bkg = df_bkg[vars(args)[roc]['truth_var']] > 0
		
		mask_sig = mask_sig & truth_mask_sig & nonzero_mask_sig
		
		
		if 'JZ' in getSamples(bkg) or 'EB' in getSamples(bkg):
			cell_mask_sig = df_sig['cell_et'] > vars(args)[roc]['cell_cut']*1000
			cell_mask_bkg = df_bkg['cell_et'] > vars(args)[roc]['cell_cut']*1000
			mask_sig = mask_sig & cell_mask_sig & nonzero_mask_sig
			mask_bkg = mask_bkg & cell_mask_bkg & nonzero_mask_bkg
		else:
			mask_bkg = mask_bkg & truth_mask_bkg & nonzero_mask_bkg 	
			
		val_sig  = df_sig[getMETAlgo(hist)][mask_sig]
		val_bkg  = df_bkg[getMETAlgo(hist)][mask_bkg]
		if vars(args)[roc]['Varunit'] == 'MeV': val_sig = val_sig / 1000
		if vars(args)[roc]['Varunit'] == 'MeV': val_bkg = val_bkg / 1000
		cuts_sig, rate_sig, low_sig, high_sig = getEfficiencies(args, val_sig, roc)
		cuts_bkg, rate_bkg, low_bkg, high_bkg = getEfficiencies(args, val_bkg, roc)	
		plt.errorbar(rate_sig, rate_bkg, fmt='-', markersize=3)
	plt.xlabel("%s (Truth MET > %s GeV)"%(vars(args)[roc]['xlabel'], vars(args)[roc]['sig_truth_cut']))
	plt.ylabel("%s (Truth MET < %s GeV)"%(vars(args)[roc]['ylabel'], vars(args)[roc]['bkg_truth_cut']))
	plt.xlim([0.75, 1.0])
	#plt.ylim([0.75, 1.0])
	plt.yscale('log')
	plt.title('model: %s, train on: %s, L1: %s, learn: %s'%(args.model_name, getSamples(args.train_on), args.L1_Trigger, getStrategy(args)))
	legend = plt.legend([getAlgoName(tag) for tag in vars(args)[roc]['vars']])
	legend.set_title("%s vs %s, L1: %s"%(getSamples(sig), getSamples(bkg) ,getTag(cut)))
	plt.savefig('%s/plots/ROC_%s_Sig_%s_Bkg_%s_model_%s_L1Cut_%s_SigTruthMETCut_%s_BkgTruthMETCut_%s.pdf'%(args.output_dir, vars(args)[roc]['name'], getSamples(sig), getSamples(bkg), args.model_name, cut, vars(args)[roc]['sig_truth_cut'],  vars(args)[roc]['bkg_truth_cut']))
	plt.close()		
			
def resos(args, data, reso):
	
	for key, item in data.items():
		L1_triggers = [col for col in item.columns if 'L1' in col]
		mask = np.ones(item.shape[0],) == 1
		resoVars(args, item, key, reso, mask, 'NONE')
		if args.doTrig:
			for L1 in L1_triggers:
				mask = item[L1] == 1
				resoVars(args, item, key, reso, mask, L1)
						
def effs(args, data, eff):

	for key, item in data.items():
		L1_triggers = [col for col in item.columns if 'L1' in col]
		mask = np.ones(item.shape[0],) == 1
		effVars(args, item, key, eff, mask, 'NONE')
		if args.doTrig:
			for L1 in L1_triggers:
				mask = item[L1] == 1
				effVars(args, item, key, eff, mask, L1)
			
def effsrate(args, data, eff):
	
	for key, item in data.items():
		
		L1_triggers = [col for col in item.columns if 'L1' in col]
		mask = np.ones(item.shape[0],) == 1
		effRateVars(args, item, key, eff, mask, 'NONE')
		if args.doTrig:
			for L1 in L1_triggers:
				mask = item[L1] == 1
				effRateVars(args, item, key, eff, mask, L1)
			
def rocs(args, data, roc): 
	
	sig = vars(args)[roc]['sig']
	bkg = vars(args)[roc]['bkg']
	df_sig = data[sig]
	df_bkg = data[bkg]
	L1_triggers = [col for col in df_sig.columns if 'L1' in col]
	mask_sig = np.ones(df_sig.shape[0],) == 1				
	mask_bkg = np.ones(df_bkg.shape[0],) == 1
	rocVars(args, df_sig, df_bkg, sig, bkg, roc, mask_sig, mask_bkg, 'NONE')
	if args.doTrig:
		for L1 in L1_triggers:
			mask_sig = df_sig[L1] == 1
			mask_bkg = df_bkg[L1] == 1
			rocVars(args, df_sig, df_bkg, sig, bkg, roc, mask_sig, mask_bkg, L1)
	
	
def computeEff(n_all, n_passed, bins):

	effs = np.array(n_passed/n_all, dtype=float)
	eff_err_l = []
	eff_err_h = []
	
	confLevel = 0.683
	for n_a, n_p in zip(n_all, n_passed):
		l, h = getEffError(n_a, n_p)
		eff_err_l.append(l)
		eff_err_h.append(h)
	
	bin_center = 0.5*(bins[1:]+bins[:-1])
	
	return bin_center, effs, np.array(eff_err_l), np.array(eff_err_h)
	
def getEffError(n_all, n_passed, confLevel = 0.683):
	l = 0
	h = 0
	eff = 0
	if n_all == 0:
		eff = 0
	else:	
		eff = float(n_passed/n_all)
	
	if n_passed == n_all:
		h = 1.0
	else: 
		h = stats.beta.ppf(confLevel + 0.5*(1-confLevel), n_passed+1., n_all-n_passed)

	if n_passed == 0:
		l = 0
	else: 
		l = stats.beta.ppf(0.5*(1+confLevel) - confLevel, n_passed, n_all-n_passed+1.)
		
	return eff-l, h-eff	
	

def computeSingleEff(n_all, n_passed):
	effs = 0
	if n_all == 0:
		effs = 0
	else:	
		effs = float(n_passed/n_all)
	l, h = getEffError(n_all, n_passed)
	return effs, l, h
		
def main():

	args = getArgs()
	model_config = args.parse_args().dir+'/model_config.json'
	args.add_argument('--config', default=model_config)
	args = add_plot_configuration(args)
	args = read_json(args)

	try:
		os.makedirs(args.output_dir+"/plots/")
	except:
		print('dir ' + args.output_dir + ' already exists')
			
	df_dic  = load_df(args)

	
	for key in args.plots.keys():
		
	
		if key == 'hist':
			print('hist: %s'%(args.plots[key]))
			for var in args.plots[key]:
				hist(args, df_dic, var)
	
		
		if key == 'stacks':
			print('stacks: %s'%(args.plots[key]))
			for stack in args.plots[key]:
				stacks(args, df_dic, stack)
		
		if key == 'profile':
			print('profiles: %s'%(args.plots[key]))
			for prof in args.plots[key]:
				profile(args, df_dic, prof)
				
		if key == 'resolution':
			print('resolution: %s'%(args.plots[key]))
			for reso in args.plots[key]:
				resos(args, df_dic, reso)
		
		'''
		if key == 'effs':
			print('effs: %s'%(args.plots[key]))
			for eff in args.plots[key]:
				effs(args, df_dic, eff)
		'''
						
		if key == 'effsrate':
			print('effs: %s'%(args.plots[key]))
			for eff in args.plots[key]:
				effsrate(args, df_dic, eff)
		
	
		if key == 'ROC':
			print('ROCs: %s'%(args.plots[key]))
			for roc in args.plots[key]:
				rocs(args, df_dic, roc)					
		
if __name__ == '__main__':
	
	main()		
