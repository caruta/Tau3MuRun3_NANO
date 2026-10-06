import subprocess, json, time, os, argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--year", required=True, type=str, help="Year of the dataset")
    argparser.add_argument("--era", type=str, default=None, help="Era of the dataset")
    argparser.add_argument("--MCera", type=str, default=None, help="MC Era of the dataset")
    argparser.add_argument("--channel", required=True, choices=["signal", "control"], help="Select 'signal' or 'control'")
    args = argparser.parse_args()
    
    year = args.year
    era = args.era
    MCera = args.MCera
    channel = args.channel

    # Definizione nome directory e file config
    suffix = era if era else MCera
    dir_name = f"{year}_era{suffix}_{channel}"
    channel_name = "Tau3mu" if channel == "signal" else "DsPhiPi"
    config_file = "test_Tau3Mu_cfg.py" if channel == "signal" else "test_DsPhiPi_cfg.py"
    
    if (MCera==None) and (era==None):
        exit()

    with open("Runs.json", "r") as file:
        data = json.load(file)[year]

    # --- case: isMC = False ---
    if era != None:
        era_index = data["Eras"].index(era) if era in data["Eras"] else None
        if era_index == None:
            print(f"WARNING: {era} is not in the list of eras. Please check.")
            exit()
            
        globaltag = data["GTs"][era_index]
        golden_json = data["golden_json"][era_index]

        with open(f"datasets/datasets_{year}_{era}.txt", "r") as filetxt:
            righe = [riga.strip() for riga in filetxt]

        command = f"""
        directory="$PWD";
        pathtoskimfile="$directory/../test"; 
        mkdir -p "{dir_name}"; 
        echo "Data {year} - era {era} - channel {channel} is selected"; 
        
        path_cfg="$directory/{dir_name}/PatAndTree_cfg.py";
        
        cp "$pathtoskimfile/{config_file}" "$path_cfg";
        
        sed -i "s#160X_dataRun3_Prompt_v1#{globaltag}#g" "$path_cfg";
        sed -i "s#options.register('isMC', True#options.register('isMC', False#g" "$path_cfg";

        cp templates/report.sh "{dir_name}/report.sh";
        cp templates/status.sh "{dir_name}/status.sh";
        cp templates/resubmit.sh "{dir_name}/resubmit.sh";
        cd "{dir_name}";
        sed -i "s#YEAR#{year}#g" *.sh;
        sed -i "s#ERANAME#{era}#g" *.sh;
        sed -i "s#CHANNELNAME#{channel_name}#g" *.sh;
        cd ..;
        cp templates/submit.sh "{dir_name}/submit.sh";
        """
        subprocess.run(command, shell=True, check=True)

        for i, dataset in enumerate(righe):
            command = f"""
            directory="$PWD";
            path="$directory/{dir_name}/PatAndTree_cfg.py";
            cp templates/CRAB_template.py "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#CHANNELNAME#{channel_name}#g" "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#YEAR#{year}#g" "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#ERANAME#{era}#g" "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#NUMBER#{i}#g" "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#DATASET_NAME#{dataset}#g" "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#FILE_TO_SUBMIT_PATH#$path#g" "{dir_name}/CRAB_stream_{i}.py";
            sed -i "s#GOLDEN_JSON_PATH#{golden_json}#g" "{dir_name}/CRAB_stream_{i}.py";
            cd "{dir_name}";
            crab submit -c "CRAB_stream_{i}.py";
            cd ..;
            """
            subprocess.run(command, shell=True, check=True)

    # --- case: isMC = True ---
    elif MCera != None:
        era_index = data["MC_era"].index(MCera) if MCera in data["MC_era"] else None
        if era_index == None:
            print(f"WARNING: {MCera} is not in the list of MC_eras. Please check.")
            exit()
            
        MC_input_type = data["MC_input_type"][era_index]
        globaltag = data["MC_GTs"][era_index]
        MC_datasets = data["MC_datasets"][era_index]


        command = f"""
        directory="$PWD";
        pathtoskimfile="$directory/../test";
        mkdir -p "{dir_name}"; 
        echo "MC {year} - {MCera} is selected"; 
        
        path_cfg="$directory/{dir_name}/PatAndTree_cfg.py";
        cp "$pathtoskimfile/{config_file}" "$path_cfg";
        
        # Replace with Global Tag MC
        sed -i "s#140X_mcRun3_2024_realistic_v26#{globaltag}#g" "$path_cfg";
        
        # Force isMC = True
        sed -i "s#options.register('isMC', False#options.register('isMC', True#g" "$path_cfg";
        """
        subprocess.run(command, shell=True, check=True)

        command = f"""
        directory="$PWD";
        path="$directory/{dir_name}/PatAndTree_cfg.py";
        cp templates/CRAB_template_MC.py "{dir_name}/CRAB_MC.py";
        sed -i "s#CHANNELNAME#{channel_name}#g" "{dir_name}/CRAB_MC.py";
        sed -i "s#YEAR#{year}#g" "{dir_name}/CRAB_MC.py";
        sed -i "s#ERANAME#{MCera}#g" "{dir_name}/CRAB_MC.py";
        sed -i "s#DATASET_NAME#{MC_datasets}#g" "{dir_name}/CRAB_MC.py";
        sed -i "s#FILE_TO_SUBMIT_PATH#$path#g" "{dir_name}/CRAB_MC.py";
        sed -i "s#INPUT_TYPE#{MC_input_type}#g" "{dir_name}/CRAB_MC.py";
        
        cd "{dir_name}";
        crab submit -c "CRAB_MC.py";
        cd ..;
        """
        subprocess.run(command, shell=True, check=True)
