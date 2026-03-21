# -*- coding: utf-8 -*-
"""
Code for 2D dynamic earthquake sequence simulation.
Written by Reiju Norisugi, Graduate School of Scienece, Kyoto Univeristy.
Last updated: 20260309.
"""

# This is the code to drive the simulation.
# This module summarize the dependence of computational conditions.
# Do not change.

import logging
import os
import platform
import subprocess
import sys
import time
import dill
import numpy as np
from sbiem_modules.quality_controll import check  # For quality control.
from sbiem_modules.utils import utils # Utility.

# Get machine type.
arch = platform.machine()

class Start:
    def __init__(self, fname, CL, RESTART, tmax_restart, dir_name):
        self.fname = fname
        self.CL = CL
        self.RESTART = RESTART # RESTART=True restarts simulation
        self.tmax_restart = tmax_restart
        self.dir_name = dir_name

    def start(self):
        # Run parameter settings.
        if not self.RESTART:
            subprocess.run(
                ["python3", "Figures{}_params_{}.py".format(self.fname, self.CL), self.dir_name]
            )
        else:
            pass

        # Load variables.
        # THis is only a place where loads variables for simulation.
        # If you change the saving format in parameter setting file, you should change here.
        Devices = utils.load("Output{}Devices.pkl".format(self.fname))
        Conditions = utils.load("Output{}Conditions.pkl".format(self.fname))
        Medium = utils.load("Output{}Medium.pkl".format(self.fname))
        FaultParams = utils.load("Output{}FaultParams.pkl".format(self.fname))
        FieldVariables = utils.load("Output{}FieldVariables.pkl".format(self.fname))
        id_station = np.load("Output{}id_station.npy".format(self.fname))

        if not self.RESTART:
            #############   Quality controll   ###############
            ch = check.Check(Devices, Conditions, Medium, FaultParams, FieldVariables)
            ch.exe()
            ##################################################

            # Measure kernel preparation time for some needs.
            self.t_prep = time.time()

            # Import module for kernel computation.
            if Conditions["G2G"]:
                if Devices["device_1"] == "cpu":
                    # For multi-CPU-core computation.
                    import sbiem_modules.convolution.conv_dyn_CPU as conv_dyn
                    import sbiem_modules.kernel.kernel_dyn_CPU as ker_dyn
                else:
                    if arch in ("x86_64"):
                        # Machine type (x86_64) allows torch.compile.
                        import sbiem_modules.convolution.conv_dyn_G2G_compile as conv_dyn
                    else:
                        # Other type (e.g., arm64) does not allow torch.compile.
                        import sbiem_modules.convolution.conv_dyn_G2G as conv_dyn
                    import sbiem_modules.kernel.kernel_dyn_G2G as ker_dyn
            else: # Computation under PCle-based system is in preparation.
                import sbiem_modules.convolution.conv_dyn_XGC as conv_dyn

            # Make class instances for kernel preparation and convolution.
            # Underscore represents instances for double-step solution.
            ker = ker_dyn.Kernel(Devices, Conditions, Medium, FaultParams, FieldVariables)
            cv = conv_dyn.Convolution(Devices, Conditions, Medium, FaultParams, FieldVariables)
            cv_ = 0 # When Stteper == 'LR' is chosen, this is set as 0. Do not delete this.
            if Conditions["Stepper"] == "RO" or Conditions["outerror"]:
                cv_ = conv_dyn.Convolution(Devices, Conditions, Medium, FaultParams, FieldVariables)

            # Prepare convolutional kernel and history.
            if Conditions["rmPB"]:
                # Compute kernel "without" periodic boundary.
                ker.comp_kernel_without_pb()
            else:
                # Compute kernel "with" periodic boundary.
                ker.comp_kernel()
            # Split and send kernels to GPUs.
            ker.chunk_tensor()
            ker.send_kernel()
            # Make history for first time step.
            cv.keep_dDell = FieldVariables["d_Dell"].clone()
            cv.store_dDell(Medium["dt0_guess"], True)
            cv.store_dDell(Medium["dt0_guess"], False)
            if Conditions["Stepper"] == "RO" or Conditions["outerror"]:
                # Make history for double stepping.
                cv_.keep_dDell = FieldVariables["d_Dell"].clone()
                cv_.store_dDell(Medium["dt0_guess"], True)
                cv_.store_dDell(Medium["dt0_guess"], False)

            # If the number of GPUs is 2 or 4, ker.kernel is split and sent to GPUs.
            # Then, ker.kernel is not used for computation.
            # For saving memory when packing the restart file, it is better to delete ker.kernel.
            if Conditions["num_GPU"] == 2 or Conditions["num_GPU"] == 4:
                ker.kernel = None

            # Solve initial stress transfer functional.
            FieldVariables["f"] = cv.exe_conv(ker,FieldVariables["Dell"],FieldVariables["d_Dell"],
                                              Medium["dt0_guess"],True,)
            if not Conditions["Frict"] == "reg_RSF_AG":
                FieldVariables["tau_ini"] = FieldVariables["tau"] - FieldVariables["f"]
            else:
                FieldVariables["tau"] += FieldVariables["f"]

            ###############   Save variables by pickle.   ################
            utils.save("Output/{}FieldVariables.pkl".format(self.fname), FieldVariables)
            ###############################################################
            self.t_prep = time.time() - self.t_prep
            np.save("Output{}t_prep.npy".format(self.fname), self.t_prep)
            ##############################################################################

            # Do not change the following section.
            ################   Import modules for given conditions   ##############
            # Import integrator, manager, adn recorder.
            if self.CL == "RSF":
                import sbiem_modules.integrator.RSF as itg
                import sbiem_modules.manager.RSF as mane
                import sbiem_modules.manager.run_RSF as run
                import sbiem_modules.record.write_RSF as write
            elif self.CL == "RRF":
                import sbiem_modules.integrator.RRF as itg
                import sbiem_modules.manager.RRF as mane
                import sbiem_modules.manager.run_RRF as run
                import sbiem_modules.record.write_RRF as write
            elif self.CL == "mRRF":
                import sbiem_modules.integrator.mRRF as itg
                import sbiem_modules.manager.mRRF as mane
                import sbiem_modules.manager.run_RRF as run
                import sbiem_modules.record.write_mRRF as write
            # For single-step solution.
            single = itg.Heun(Devices, Conditions, Medium, FaultParams, FieldVariables)
            double = 0 # Do not delete this.
            if Conditions["Stepper"] == "RO" or Conditions["outerror"]:
                # For double-step solution.
                double = itg.Heun(Devices, Conditions, Medium, FaultParams, FieldVariables)
            # Make class instances for recording output, managing simulation, and running time loop.
            wr = write.Write(self.fname, Devices, Conditions, Medium, FaultParams, FieldVariables)
            st = mane.Management(Devices, Conditions, Medium, FaultParams, FieldVariables, id_station)
            sim = run.Simulate(Devices, Conditions, Medium, FaultParams, FieldVariables)

            # Make class instances for time stepper.
            if Conditions["Stepper"] == "RO":
                import sbiem_modules.time_step.RO as RO
                ti = RO.Time_RO(Devices, Conditions, Medium, FaultParams, FieldVariables)
            elif Conditions["Stepper"] == "LR":
                import sbiem_modules.time_step.LR as LR
                ti = LR.Time_LR(Devices, Conditions, Medium, FaultParams, FieldVariables)
            elif Conditions["Stepper"] == "CS":
                import sbiem_modules.time_step.CS as CS
                ti = CS.Time_CS(Devices, Conditions, Medium, FaultParams, FieldVariables)

            # Import solver for each constitutive law.
            if Conditions["Frict"] == "RSF_AG":
                import sbiem_modules.constitutive_law.RSF_AG as cl
            elif Conditions["Frict"] == "reg_RSF_AG":
                import sbiem_modules.constitutive_law.reg_RSF_AG as cl
            elif Conditions["Frict"] == "RRF":
                import sbiem_modules.constitutive_law.RRF as cl
            elif Conditions["Frict"] == "mRRF" and Conditions["Cutoff"]:
                import sbiem_modules.constitutive_law.mRRFcoV as cl
            elif Conditions["Frict"] == "mRRF" and not Conditions["Cutoff"]:
                import sbiem_modules.constitutive_law.mRRF as cl
            # Make class instances for solver.
            # For single-step solution.
            pr = cl.Solver(Devices, Conditions, Medium, FaultParams, FieldVariables)
            update = cl.Update(Devices, Conditions, Medium, FaultParams, FieldVariables)
            pr_ = 0      # Do not delete this.
            update_ = 0  # Do not delete this.
            if Conditions["Stepper"] == "RO" or Conditions["outerror"]:
                # For single-step solution.
                pr_ = cl.Solver(Devices, Conditions, Medium, FaultParams, FieldVariables)
                update_ = cl.Update(Devices, Conditions, Medium, FaultParams, FieldVariables)
            ########################################################################

            ###############   Check initial conditions.   ################
            if Conditions["Initial"]:
                os.makedirs("Figures{}Initial".format(self.fname), exist_ok=True)
                if self.CL == "RSF":
                    import sbiem_modules.visualize_initial.visualize_RSF as visualize
                elif self.CL == "RRF":
                    import sbiem_modules.visualize_initial.visualize_RRF as visualize
                elif self.CL == "mRRF":
                    import sbiem_modules.visualize_initial.visualize_mRRF as visualize
                vis = visualize.Vis(self.fname, Devices, Conditions, Medium,
                                    FaultParams, FieldVariables, id_station)
                # Plot figures to show the initial conditions.
                vis.figs()
                logging.info("")
                logging.info("Check initial conditions.")
                sys.exit(0)
            else:
                pass
            ##############################################################

            ##########################   Running Simulation   ###########################
            # Open output files.
            wr.open_files()

            # Initialize strage variables.
            st.init_strage()

            # Solve first time step size.
            st.time_upt(ti)

            # Run simulations.
            # Total computational time is measured for some needs.
            self.t_wh = time.time()
            if Conditions["Stepper"] == "RO":
                st.store_RO(double, sim.error, ker)  # Record initial conditions.
                # Main loop.
                sim.loop_RO(single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti)
            elif Conditions["Stepper"] == "LR":
                st.store_LR(single, sim.error, ker)  # Record initial conditions.
                # Main loop.
                sim.loop_LR(single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti)
            elif Conditions["Stepper"] == "CS":
                st.store_LR(single, ker)  # Record initial conditions.
                # Main loop.
                sim.loop_CS(single, st, ker, cv, pr, update, wr, ti)
            ##########################################################################

            ###########    Save variables again by pickle as ndarray.   ##################
            # Somehow I need this step to analize the result on CPU after computation.
            FaultParams = {k: utils.ttn(v) for k, v in FaultParams.items()}
            utils.save("Output{}FaultParams.pkl".format(self.fname), FaultParams)
            ###############################################################################
        else:
            #######################  Section for restart.  ################################
            # Load class instances by dill for restarting.
            with open("Restart{}restart.pkl".format(self.fname), "rb") as file:
                data = dill.load(file)
                st = data["st"]
                wr = data["wr"]
                ti = data["ti"]
                ker = data["ker"]
                update = data["update"]
                update_ = data["update_"]
                cv = data["cv"]
                cv_ = data["cv_"]
                pr = data["pr"]
                pr_ = data["pr_"]
                single = data["single"]
                double = data["double"]
                sim = data["sim"]
                data = None

            # Set tmax for restarted simulation.
            sim.tmax = self.tmax_restart

            # Open output files.
            wr.open_files()

            # Re-create kernel and history. Do not change.
            if Conditions["Stepper"] == "RO":
                # Prepare kernel.
                if ker.rmPB:
                    ker.comp_kernel_without_pb()
                else:
                    ker.comp_kernel()
                # Split kernel for multiple GPUs.
                ker.chunk_tensor()
                ker.send_kernel()

                # Re-create history.
                if (
                    cv_.previously_dt_ge_Tw
                ):  # This is the case when dt > truncation window.
                    # Create empty history on CPU.
                    cv_.create_hist_on_cpu()
                    # Send them to GPU.
                    cv_.send_history()
                    # Re-construct history by cv.kkeep_dDell.
                    cv_.copy_all_hist()
                else:
                    # If previous dt < Tw, the whole history was saved.
                    # You can just send it to GPU.
                    cv_.send_history()
                if not ker.qd:
                    # Clone histories.
                    if cv.split:
                        if cv.num_GPU == 0 or cv.num_GPU == 1:
                            cv.dDell_hist_r = cv_.dDell_hist_r.clone()
                            cv.dDell_hist_i = cv_.dDell_hist_i.clone()
                        elif cv.num_GPU == 2:
                            cv.dDell_hist_1r = cv_.dDell_hist_1r.clone()
                            cv.dDell_hist_1i = cv_.dDell_hist_1i.clone()
                            cv.dDell_hist_2r = cv_.dDell_hist_2r.clone()
                            cv.dDell_hist_2i = cv_.dDell_hist_2i.clone()
                            cv.kernel = None
                            cv_.kernel = None
                        elif cv.num_GPU == 4:
                            cv.dDell_hist_1r = cv_.dDell_hist_1r.clone()
                            cv.dDell_hist_1i = cv_.dDell_hist_1i.clone()
                            cv.dDell_hist_2r = cv_.dDell_hist_2r.clone()
                            cv.dDell_hist_2i = cv_.dDell_hist_2i.clone()
                            cv.dDell_hist_3r = cv_.dDell_hist_3r.clone()
                            cv.dDell_hist_3i = cv_.dDell_hist_3i.clone()
                            cv.dDell_hist_4r = cv_.dDell_hist_4r.clone()
                            cv.dDell_hist_4i = cv_.dDell_hist_4i.clone()
                            cv.kernel = None
                            cv_.kernel = None
                    else:
                        if cv.num_GPU == 0 or cv.num_GPU == 1:
                            cv.dDell_hist = cv_.dDell_hist.clone()
                        elif cv.num_GPU == 2:
                            cv.dDell_hist_1 = cv_.dDell_hist_1.clone()
                            cv.dDell_hist_2 = cv_.dDell_hist_2.clone()
                            cv.kernel = None
                            cv_.kernel = None
                        elif cv.num_GPU == 4:
                            cv.dDell_hist_1 = cv_.dDell_hist_1.clone()
                            cv.dDell_hist_2 = cv_.dDell_hist_2.clone()
                            cv.dDell_hist_3 = cv_.dDell_hist_3.clone()
                            cv.dDell_hist_4 = cv_.dDell_hist_4.clone()
                            cv.kernel = None
                            cv_.kernel = None
            else:
                if not ker.qd:
                    if ker.rmPB:
                        ker.comp_kernel_without_pb()
                    else:
                        ker.comp_kernel()
                    ker.chunk_tensor()
                    ker.send_kernel()
                    if cv.previously_dt_ge_Tw:
                        cv.create_hist_on_cpu()
                        cv.send_history()
                        cv.copy_all_hist()
                    else:
                        cv.send_history()

            # Restart simulation.
            self.t_wh = time.time()
            if Conditions["Stepper"] == "RO":
                sim.loop_RO(single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti)
            elif Conditions["Stepper"] == "LR":
                sim.loop_LR(single, double, st, ker, cv, cv_, pr, pr_, update, update_, wr, ti)
            elif Conditions["Stepper"] == "CS":
                sim.loop_CS(single, st, ker, cv, pr, update, wr, ti)
            ##################################################################################

        ##################   Section to wrap-up the simulation results.   ####################
        # Write results holded by storage variables and close files.
        st.exe_write(wr)
        wr.close_files()
        if Conditions["act_res"]:
            # For writing results additionally after restarting, writing mode is changed.
            wr.mode = "ab"
            wr.open_files()
            wr.close_files()

            # Delete kernels and history for saving storage.
            # Note that the kernel is re-computed when restarting.
            if Conditions["Stepper"] == "RO":
                # Delete kernel for single step.
                ker.reset_kernel()
                # Delete history for single step.
                cv.reset_history()
                if cv_.previously_dt_ge_Tw:
                    # When dt > Tw, cv_.kkeep_dDell can hold full information of history.
                    # The history can be deleted here.
                    cv_.reset_history()
                else:
                    # When dt < Tw, history is complex. Full history is saved.
                    # If you finish the simulation by setting wall time, this can happen frequently.
                    # The size of restart file will be huge.
                    cv_.sendback_history()
            else:
                ker.reset_kernel()
                if cv.previously_dt_ge_Tw:
                    cv.reset_history()
                else:
                    cv.sendback_history()

        # Save counting variables.
        np.save("Output{}id_count.npy".format(self.fname), st.id_count)
        np.save("Output{}id_count_st.npy".format(self.fname), st.id_count_st)
        np.save("Output{}EQ_num.npy".format(self.fname), st.EQ_num)

        ####################  Informations.  #######################
        self.t_wh = time.time() - self.t_wh
        day2sec = 24.0 * 60.0 * 60.0
        min2sec = 60.0
        np.save("Output{}t_wh.npy".format(self.fname), self.t_wh)
        logging.info("")
        logging.info("*************************************************")
        logging.info("")
        if self.t_wh / day2sec >= 0.1:
            logging.info(f"Consuming time {self.t_wh / day2sec} day")
        elif self.t_wh / day2sec < 0.1 and self.t_wh / min2sec >= 5.0:
            logging.info(f"Consuming time {self.t_wh / min2sec} min")
        else:
            logging.info(f"Consuming time {self.t_wh} sec")
        logging.info(f"{st.id_count_st} time steps")
        logging.info(f"{st.EQ_num} quakes")
        logging.info("")
        ############################################################

        ################  Save all the variables by dill for restart  ###################
        if Conditions["act_res"]:
            with open("Restart{}restart.pkl".format(self.fname), "wb") as file:
                dill.dump(
                    {
                        "st": st,
                        "wr": wr,
                        "ti": ti,
                        "ker": ker,
                        "update": update,
                        "update_": update_,
                        "cv": cv,
                        "cv_": cv_,
                        "pr": pr,
                        "pr_": pr_,
                        "single": single,
                        "double": double,
                        "sim": sim,
                    },
                    file,
                )
            logging.info("Restart files are saved.")
            sys.exit(0)
        #################################################################################
        ################################################################################################
