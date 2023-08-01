---
layout: default
title: maml.apps.md
nav_exclude: true
---

# maml.apps package

## Subpackages


* [maml.apps.bowsr package](maml.apps.bowsr.md)


    * [Subpackages](maml.apps.bowsr.md#subpackages)


        * [maml.apps.bowsr.model package](maml.apps.bowsr.model.md)


            * [`EnergyModel`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.EnergyModel)


                * [`EnergyModel.predict_energy()`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.EnergyModel.predict_energy)




            * [maml.apps.bowsr.model.base module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.base)


                * [`EnergyModel`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.base.EnergyModel)


                    * [`EnergyModel.predict_energy()`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.base.EnergyModel.predict_energy)


            * [maml.apps.bowsr.model.cgcnn module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.cgcnn)


            * [maml.apps.bowsr.model.dft module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.dft)


                * [`DFT`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.dft.DFT)


                    * [`DFT.predict_energy()`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.dft.DFT.predict_energy)


            * [maml.apps.bowsr.model.megnet module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.megnet)




    * [maml.apps.bowsr.acquisition module](maml.apps.bowsr.md#module-maml.apps.bowsr.acquisition)


        * [`AcquisitionFunction`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.AcquisitionFunction)


            * [`AcquisitionFunction._ei()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.AcquisitionFunction._ei)


            * [`AcquisitionFunction._gpucb()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.AcquisitionFunction._gpucb)


            * [`AcquisitionFunction._poi()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.AcquisitionFunction._poi)


            * [`AcquisitionFunction._ucb()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.AcquisitionFunction._ucb)


            * [`AcquisitionFunction.calculate()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.AcquisitionFunction.calculate)


        * [`_trunc()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition._trunc)


        * [`ensure_rng()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.ensure_rng)


        * [`lhs_sample()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.lhs_sample)


        * [`predict_mean_std()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.predict_mean_std)


        * [`propose_query_point()`](maml.apps.bowsr.md#maml.apps.bowsr.acquisition.propose_query_point)


    * [maml.apps.bowsr.optimizer module](maml.apps.bowsr.md#module-maml.apps.bowsr.optimizer)


        * [`BayesianOptimizer`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer)


            * [`BayesianOptimizer.add_query()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.add_query)


            * [`BayesianOptimizer.as_dict()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.as_dict)


            * [`BayesianOptimizer.from_dict()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.from_dict)


            * [`BayesianOptimizer.get_derived_structure()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.get_derived_structure)


            * [`BayesianOptimizer.get_formation_energy()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.get_formation_energy)


            * [`BayesianOptimizer.get_optimized_structure_and_energy()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.get_optimized_structure_and_energy)


            * [`BayesianOptimizer.gpr`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.gpr)


            * [`BayesianOptimizer.optimize()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.optimize)


            * [`BayesianOptimizer.propose()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.propose)


            * [`BayesianOptimizer.set_bounds()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.set_bounds)


            * [`BayesianOptimizer.set_gpr_params()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.set_gpr_params)


            * [`BayesianOptimizer.set_space_empty()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.set_space_empty)


            * [`BayesianOptimizer.space`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.BayesianOptimizer.space)


        * [`atoms_crowded()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.atoms_crowded)


        * [`struct2perturbation()`](maml.apps.bowsr.md#maml.apps.bowsr.optimizer.struct2perturbation)


    * [maml.apps.bowsr.perturbation module](maml.apps.bowsr.md#module-maml.apps.bowsr.perturbation)


        * [`LatticePerturbation`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.LatticePerturbation)


            * [`LatticePerturbation.abc`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.LatticePerturbation.abc)


            * [`LatticePerturbation.fit_lattice`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.LatticePerturbation.fit_lattice)


            * [`LatticePerturbation.lattice`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.LatticePerturbation.lattice)


            * [`LatticePerturbation.sanity_check()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.LatticePerturbation.sanity_check)


        * [`WyckoffPerturbation`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.WyckoffPerturbation)


            * [`WyckoffPerturbation.fit_site`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.WyckoffPerturbation.fit_site)


            * [`WyckoffPerturbation.get_orbit()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.WyckoffPerturbation.get_orbit)


            * [`WyckoffPerturbation.sanity_check()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.WyckoffPerturbation.sanity_check)


            * [`WyckoffPerturbation.site`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.WyckoffPerturbation.site)


            * [`WyckoffPerturbation.standardize()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.WyckoffPerturbation.standardize)


        * [`crystal_system()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.crystal_system)


        * [`get_standardized_structure()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.get_standardized_structure)


        * [`perturbation_mapping()`](maml.apps.bowsr.md#maml.apps.bowsr.perturbation.perturbation_mapping)


    * [maml.apps.bowsr.preprocessing module](maml.apps.bowsr.md#module-maml.apps.bowsr.preprocessing)


        * [`DummyScaler`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.DummyScaler)


            * [`DummyScaler.as_dict()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.DummyScaler.as_dict)


            * [`DummyScaler.fit()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.DummyScaler.fit)


            * [`DummyScaler.from_dict()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.DummyScaler.from_dict)


            * [`DummyScaler.inverse_transform()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.DummyScaler.inverse_transform)


            * [`DummyScaler.transform()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.DummyScaler.transform)


        * [`StandardScaler`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.StandardScaler)


            * [`StandardScaler.as_dict()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.StandardScaler.as_dict)


            * [`StandardScaler.fit()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.StandardScaler.fit)


            * [`StandardScaler.from_dict()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.StandardScaler.from_dict)


            * [`StandardScaler.inverse_transform()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.StandardScaler.inverse_transform)


            * [`StandardScaler.transform()`](maml.apps.bowsr.md#maml.apps.bowsr.preprocessing.StandardScaler.transform)


    * [maml.apps.bowsr.target_space module](maml.apps.bowsr.md#module-maml.apps.bowsr.target_space)


        * [`TargetSpace`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace)


            * [`TargetSpace.bounds`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.bounds)


            * [`TargetSpace.lhs_sample()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.lhs_sample)


            * [`TargetSpace.params`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.params)


            * [`TargetSpace.probe()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.probe)


            * [`TargetSpace.register()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.register)


            * [`TargetSpace.set_bounds()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.set_bounds)


            * [`TargetSpace.set_empty()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.set_empty)


            * [`TargetSpace.target`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.target)


            * [`TargetSpace.uniform_sample()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space.TargetSpace.uniform_sample)


        * [`_hashable()`](maml.apps.bowsr.md#maml.apps.bowsr.target_space._hashable)


* [maml.apps.gbe package](maml.apps.gbe.md)




    * [maml.apps.gbe.describer module](maml.apps.gbe.md#module-maml.apps.gbe.describer)


        * [`GBBond`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond)


            * [`GBBond.NNDict`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.NNDict)


            * [`GBBond._get_bond_mat()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond._get_bond_mat)


            * [`GBBond.as_dict()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.as_dict)


            * [`GBBond.bond_matrix`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.bond_matrix)


            * [`GBBond.from_dict()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.from_dict)


            * [`GBBond.get_breakbond_ratio()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.get_breakbond_ratio)


            * [`GBBond.get_mean_bl_chg()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.get_mean_bl_chg)


            * [`GBBond.max_bl`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.max_bl)


            * [`GBBond.min_bl`](maml.apps.gbe.md#maml.apps.gbe.describer.GBBond.min_bl)


        * [`GBDescriber`](maml.apps.gbe.md#maml.apps.gbe.describer.GBDescriber)


            * [`GBDescriber._abc_impl`](maml.apps.gbe.md#maml.apps.gbe.describer.GBDescriber._abc_impl)


            * [`GBDescriber._sklearn_auto_wrap_output_keys`](maml.apps.gbe.md#maml.apps.gbe.describer.GBDescriber._sklearn_auto_wrap_output_keys)


            * [`GBDescriber.generate_bulk_ref()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBDescriber.generate_bulk_ref)


            * [`GBDescriber.transform_one()`](maml.apps.gbe.md#maml.apps.gbe.describer.GBDescriber.transform_one)


        * [`convert_hcp_direction()`](maml.apps.gbe.md#maml.apps.gbe.describer.convert_hcp_direction)


        * [`convert_hcp_plane()`](maml.apps.gbe.md#maml.apps.gbe.describer.convert_hcp_plane)


        * [`get_elemental_feature()`](maml.apps.gbe.md#maml.apps.gbe.describer.get_elemental_feature)


        * [`get_structural_feature()`](maml.apps.gbe.md#maml.apps.gbe.describer.get_structural_feature)


    * [maml.apps.gbe.presetfeatures module](maml.apps.gbe.md#module-maml.apps.gbe.presetfeatures)


        * [`my_quant`](maml.apps.gbe.md#maml.apps.gbe.presetfeatures.my_quant)


            * [`my_quant.latex`](maml.apps.gbe.md#maml.apps.gbe.presetfeatures.my_quant.latex)


            * [`my_quant.name`](maml.apps.gbe.md#maml.apps.gbe.presetfeatures.my_quant.name)


            * [`my_quant.unit`](maml.apps.gbe.md#maml.apps.gbe.presetfeatures.my_quant.unit)


    * [maml.apps.gbe.utils module](maml.apps.gbe.md#module-maml.apps.gbe.utils)


        * [`load_b0_dict()`](maml.apps.gbe.md#maml.apps.gbe.utils.load_b0_dict)


        * [`load_data()`](maml.apps.gbe.md#maml.apps.gbe.utils.load_data)


        * [`load_mean_delta_bl_dict()`](maml.apps.gbe.md#maml.apps.gbe.utils.load_mean_delta_bl_dict)


        * [`update_b0_dict()`](maml.apps.gbe.md#maml.apps.gbe.utils.update_b0_dict)


* [maml.apps.pes package](maml.apps.pes.md)


    * [`DefectFormation`](maml.apps.pes.md#maml.apps.pes.DefectFormation)


        * [`DefectFormation._parse()`](maml.apps.pes.md#maml.apps.pes.DefectFormation._parse)


        * [`DefectFormation._sanity_check()`](maml.apps.pes.md#maml.apps.pes.DefectFormation._sanity_check)


        * [`DefectFormation._setup()`](maml.apps.pes.md#maml.apps.pes.DefectFormation._setup)


        * [`DefectFormation.calculate()`](maml.apps.pes.md#maml.apps.pes.DefectFormation.calculate)


        * [`DefectFormation.get_unit_cell()`](maml.apps.pes.md#maml.apps.pes.DefectFormation.get_unit_cell)


    * [`ElasticConstant`](maml.apps.pes.md#maml.apps.pes.ElasticConstant)


        * [`ElasticConstant._RESTART_CONFIG`](maml.apps.pes.md#maml.apps.pes.ElasticConstant._RESTART_CONFIG)


        * [`ElasticConstant._parse()`](maml.apps.pes.md#maml.apps.pes.ElasticConstant._parse)


        * [`ElasticConstant._sanity_check()`](maml.apps.pes.md#maml.apps.pes.ElasticConstant._sanity_check)


        * [`ElasticConstant._setup()`](maml.apps.pes.md#maml.apps.pes.ElasticConstant._setup)


    * [`EnergyForceStress`](maml.apps.pes.md#maml.apps.pes.EnergyForceStress)


        * [`EnergyForceStress._parse()`](maml.apps.pes.md#maml.apps.pes.EnergyForceStress._parse)


        * [`EnergyForceStress._rotate_force_stress()`](maml.apps.pes.md#maml.apps.pes.EnergyForceStress._rotate_force_stress)


        * [`EnergyForceStress._sanity_check()`](maml.apps.pes.md#maml.apps.pes.EnergyForceStress._sanity_check)


        * [`EnergyForceStress._setup()`](maml.apps.pes.md#maml.apps.pes.EnergyForceStress._setup)


        * [`EnergyForceStress.calculate()`](maml.apps.pes.md#maml.apps.pes.EnergyForceStress.calculate)


    * [`GAPotential`](maml.apps.pes.md#maml.apps.pes.GAPotential)


        * [`GAPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes.GAPotential._abc_impl)


        * [`GAPotential._line_up()`](maml.apps.pes.md#maml.apps.pes.GAPotential._line_up)


        * [`GAPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes.GAPotential.evaluate)


        * [`GAPotential.from_config()`](maml.apps.pes.md#maml.apps.pes.GAPotential.from_config)


        * [`GAPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes.GAPotential.pair_coeff)


        * [`GAPotential.pair_style`](maml.apps.pes.md#maml.apps.pes.GAPotential.pair_style)


        * [`GAPotential.read_cfgs()`](maml.apps.pes.md#maml.apps.pes.GAPotential.read_cfgs)


        * [`GAPotential.save()`](maml.apps.pes.md#maml.apps.pes.GAPotential.save)


        * [`GAPotential.train()`](maml.apps.pes.md#maml.apps.pes.GAPotential.train)


        * [`GAPotential.write_cfgs()`](maml.apps.pes.md#maml.apps.pes.GAPotential.write_cfgs)


        * [`GAPotential.write_param()`](maml.apps.pes.md#maml.apps.pes.GAPotential.write_param)


    * [`LMPStaticCalculator`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator)


        * [`LMPStaticCalculator._COMMON_CMDS`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator._COMMON_CMDS)


        * [`LMPStaticCalculator._parse()`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator._parse)


        * [`LMPStaticCalculator._sanity_check()`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator._sanity_check)


        * [`LMPStaticCalculator._setup()`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator._setup)


        * [`LMPStaticCalculator.allowed_kwargs`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator.allowed_kwargs)


        * [`LMPStaticCalculator.calculate()`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator.calculate)


        * [`LMPStaticCalculator.set_lmp_exe()`](maml.apps.pes.md#maml.apps.pes.LMPStaticCalculator.set_lmp_exe)


    * [`LatticeConstant`](maml.apps.pes.md#maml.apps.pes.LatticeConstant)


        * [`LatticeConstant.calculate()`](maml.apps.pes.md#maml.apps.pes.LatticeConstant.calculate)


    * [`MTPotential`](maml.apps.pes.md#maml.apps.pes.MTPotential)


        * [`MTPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes.MTPotential._abc_impl)


        * [`MTPotential._line_up()`](maml.apps.pes.md#maml.apps.pes.MTPotential._line_up)


        * [`MTPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes.MTPotential.evaluate)


        * [`MTPotential.from_config()`](maml.apps.pes.md#maml.apps.pes.MTPotential.from_config)


        * [`MTPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes.MTPotential.pair_coeff)


        * [`MTPotential.pair_style`](maml.apps.pes.md#maml.apps.pes.MTPotential.pair_style)


        * [`MTPotential.read_cfgs()`](maml.apps.pes.md#maml.apps.pes.MTPotential.read_cfgs)


        * [`MTPotential.train()`](maml.apps.pes.md#maml.apps.pes.MTPotential.train)


        * [`MTPotential.write_cfg()`](maml.apps.pes.md#maml.apps.pes.MTPotential.write_cfg)


        * [`MTPotential.write_ini()`](maml.apps.pes.md#maml.apps.pes.MTPotential.write_ini)


        * [`MTPotential.write_param()`](maml.apps.pes.md#maml.apps.pes.MTPotential.write_param)


    * [`NNPotential`](maml.apps.pes.md#maml.apps.pes.NNPotential)


        * [`NNPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes.NNPotential._abc_impl)


        * [`NNPotential._line_up()`](maml.apps.pes.md#maml.apps.pes.NNPotential._line_up)


        * [`NNPotential.bohr_to_angstrom`](maml.apps.pes.md#maml.apps.pes.NNPotential.bohr_to_angstrom)


        * [`NNPotential.eV_to_Ha`](maml.apps.pes.md#maml.apps.pes.NNPotential.eV_to_Ha)


        * [`NNPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes.NNPotential.evaluate)


        * [`NNPotential.from_config()`](maml.apps.pes.md#maml.apps.pes.NNPotential.from_config)


        * [`NNPotential.load_input()`](maml.apps.pes.md#maml.apps.pes.NNPotential.load_input)


        * [`NNPotential.load_scaler()`](maml.apps.pes.md#maml.apps.pes.NNPotential.load_scaler)


        * [`NNPotential.load_weights()`](maml.apps.pes.md#maml.apps.pes.NNPotential.load_weights)


        * [`NNPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes.NNPotential.pair_coeff)


        * [`NNPotential.pair_style`](maml.apps.pes.md#maml.apps.pes.NNPotential.pair_style)


        * [`NNPotential.read_cfgs()`](maml.apps.pes.md#maml.apps.pes.NNPotential.read_cfgs)


        * [`NNPotential.train()`](maml.apps.pes.md#maml.apps.pes.NNPotential.train)


        * [`NNPotential.write_cfgs()`](maml.apps.pes.md#maml.apps.pes.NNPotential.write_cfgs)


        * [`NNPotential.write_input()`](maml.apps.pes.md#maml.apps.pes.NNPotential.write_input)


        * [`NNPotential.write_param()`](maml.apps.pes.md#maml.apps.pes.NNPotential.write_param)


    * [`NudgedElasticBand`](maml.apps.pes.md#maml.apps.pes.NudgedElasticBand)


        * [`NudgedElasticBand._parse()`](maml.apps.pes.md#maml.apps.pes.NudgedElasticBand._parse)


        * [`NudgedElasticBand._sanity_check()`](maml.apps.pes.md#maml.apps.pes.NudgedElasticBand._sanity_check)


        * [`NudgedElasticBand._setup()`](maml.apps.pes.md#maml.apps.pes.NudgedElasticBand._setup)


        * [`NudgedElasticBand.calculate()`](maml.apps.pes.md#maml.apps.pes.NudgedElasticBand.calculate)


        * [`NudgedElasticBand.get_unit_cell()`](maml.apps.pes.md#maml.apps.pes.NudgedElasticBand.get_unit_cell)


    * [`Potential`](maml.apps.pes.md#maml.apps.pes.Potential)


    * [`SNAPotential`](maml.apps.pes.md#maml.apps.pes.SNAPotential)


        * [`SNAPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes.SNAPotential._abc_impl)


        * [`SNAPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes.SNAPotential.evaluate)


        * [`SNAPotential.from_config()`](maml.apps.pes.md#maml.apps.pes.SNAPotential.from_config)


        * [`SNAPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes.SNAPotential.pair_coeff)


        * [`SNAPotential.pair_style`](maml.apps.pes.md#maml.apps.pes.SNAPotential.pair_style)


        * [`SNAPotential.train()`](maml.apps.pes.md#maml.apps.pes.SNAPotential.train)


        * [`SNAPotential.write_param()`](maml.apps.pes.md#maml.apps.pes.SNAPotential.write_param)


    * [`SpectralNeighborAnalysis`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis)


        * [`SpectralNeighborAnalysis._CMDS`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis._CMDS)


        * [`SpectralNeighborAnalysis._parse()`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis._parse)


        * [`SpectralNeighborAnalysis._sanity_check()`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis._sanity_check)


        * [`SpectralNeighborAnalysis._setup()`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis._setup)


        * [`SpectralNeighborAnalysis.get_bs_subscripts()`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis.get_bs_subscripts)


        * [`SpectralNeighborAnalysis.n_bs`](maml.apps.pes.md#maml.apps.pes.SpectralNeighborAnalysis.n_bs)


    * [`get_default_lmp_exe()`](maml.apps.pes.md#maml.apps.pes.get_default_lmp_exe)




    * [maml.apps.pes._base module](maml.apps.pes.md#module-maml.apps.pes._base)


        * [`Potential`](maml.apps.pes.md#maml.apps.pes._base.Potential)


        * [`PotentialMixin`](maml.apps.pes.md#maml.apps.pes._base.PotentialMixin)


            * [`PotentialMixin.evaluate()`](maml.apps.pes.md#maml.apps.pes._base.PotentialMixin.evaluate)


            * [`PotentialMixin.from_config()`](maml.apps.pes.md#maml.apps.pes._base.PotentialMixin.from_config)


            * [`PotentialMixin.predict_efs()`](maml.apps.pes.md#maml.apps.pes._base.PotentialMixin.predict_efs)


            * [`PotentialMixin.train()`](maml.apps.pes.md#maml.apps.pes._base.PotentialMixin.train)


            * [`PotentialMixin.write_param()`](maml.apps.pes.md#maml.apps.pes._base.PotentialMixin.write_param)


    * [maml.apps.pes._gap module](maml.apps.pes.md#module-maml.apps.pes._gap)


        * [`GAPotential`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential)


            * [`GAPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential._abc_impl)


            * [`GAPotential._line_up()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential._line_up)


            * [`GAPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.evaluate)


            * [`GAPotential.from_config()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.from_config)


            * [`GAPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.pair_coeff)


            * [`GAPotential.pair_style`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.pair_style)


            * [`GAPotential.read_cfgs()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.read_cfgs)


            * [`GAPotential.save()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.save)


            * [`GAPotential.train()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.train)


            * [`GAPotential.write_cfgs()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.write_cfgs)


            * [`GAPotential.write_param()`](maml.apps.pes.md#maml.apps.pes._gap.GAPotential.write_param)


    * [maml.apps.pes._lammps module](maml.apps.pes.md#module-maml.apps.pes._lammps)


        * [`DefectFormation`](maml.apps.pes.md#maml.apps.pes._lammps.DefectFormation)


            * [`DefectFormation._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.DefectFormation._parse)


            * [`DefectFormation._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.DefectFormation._sanity_check)


            * [`DefectFormation._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.DefectFormation._setup)


            * [`DefectFormation.calculate()`](maml.apps.pes.md#maml.apps.pes._lammps.DefectFormation.calculate)


            * [`DefectFormation.get_unit_cell()`](maml.apps.pes.md#maml.apps.pes._lammps.DefectFormation.get_unit_cell)


        * [`ElasticConstant`](maml.apps.pes.md#maml.apps.pes._lammps.ElasticConstant)


            * [`ElasticConstant._RESTART_CONFIG`](maml.apps.pes.md#maml.apps.pes._lammps.ElasticConstant._RESTART_CONFIG)


            * [`ElasticConstant._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.ElasticConstant._parse)


            * [`ElasticConstant._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.ElasticConstant._sanity_check)


            * [`ElasticConstant._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.ElasticConstant._setup)


        * [`EnergyForceStress`](maml.apps.pes.md#maml.apps.pes._lammps.EnergyForceStress)


            * [`EnergyForceStress._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.EnergyForceStress._parse)


            * [`EnergyForceStress._rotate_force_stress()`](maml.apps.pes.md#maml.apps.pes._lammps.EnergyForceStress._rotate_force_stress)


            * [`EnergyForceStress._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.EnergyForceStress._sanity_check)


            * [`EnergyForceStress._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.EnergyForceStress._setup)


            * [`EnergyForceStress.calculate()`](maml.apps.pes.md#maml.apps.pes._lammps.EnergyForceStress.calculate)


        * [`LMPRelaxationCalculator`](maml.apps.pes.md#maml.apps.pes._lammps.LMPRelaxationCalculator)


            * [`LMPRelaxationCalculator._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPRelaxationCalculator._parse)


            * [`LMPRelaxationCalculator._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPRelaxationCalculator._sanity_check)


            * [`LMPRelaxationCalculator._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPRelaxationCalculator._setup)


        * [`LMPStaticCalculator`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator)


            * [`LMPStaticCalculator._COMMON_CMDS`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator._COMMON_CMDS)


            * [`LMPStaticCalculator._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator._parse)


            * [`LMPStaticCalculator._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator._sanity_check)


            * [`LMPStaticCalculator._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator._setup)


            * [`LMPStaticCalculator.allowed_kwargs`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator.allowed_kwargs)


            * [`LMPStaticCalculator.calculate()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator.calculate)


            * [`LMPStaticCalculator.set_lmp_exe()`](maml.apps.pes.md#maml.apps.pes._lammps.LMPStaticCalculator.set_lmp_exe)


        * [`LammpsPotential`](maml.apps.pes.md#maml.apps.pes._lammps.LammpsPotential)


            * [`LammpsPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes._lammps.LammpsPotential._abc_impl)


            * [`LammpsPotential.predict_efs()`](maml.apps.pes.md#maml.apps.pes._lammps.LammpsPotential.predict_efs)


        * [`LatticeConstant`](maml.apps.pes.md#maml.apps.pes._lammps.LatticeConstant)


            * [`LatticeConstant.calculate()`](maml.apps.pes.md#maml.apps.pes._lammps.LatticeConstant.calculate)


        * [`NudgedElasticBand`](maml.apps.pes.md#maml.apps.pes._lammps.NudgedElasticBand)


            * [`NudgedElasticBand._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.NudgedElasticBand._parse)


            * [`NudgedElasticBand._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.NudgedElasticBand._sanity_check)


            * [`NudgedElasticBand._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.NudgedElasticBand._setup)


            * [`NudgedElasticBand.calculate()`](maml.apps.pes.md#maml.apps.pes._lammps.NudgedElasticBand.calculate)


            * [`NudgedElasticBand.get_unit_cell()`](maml.apps.pes.md#maml.apps.pes._lammps.NudgedElasticBand.get_unit_cell)


        * [`SpectralNeighborAnalysis`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis)


            * [`SpectralNeighborAnalysis._CMDS`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis._CMDS)


            * [`SpectralNeighborAnalysis._parse()`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis._parse)


            * [`SpectralNeighborAnalysis._sanity_check()`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis._sanity_check)


            * [`SpectralNeighborAnalysis._setup()`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis._setup)


            * [`SpectralNeighborAnalysis.get_bs_subscripts()`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis.get_bs_subscripts)


            * [`SpectralNeighborAnalysis.n_bs`](maml.apps.pes.md#maml.apps.pes._lammps.SpectralNeighborAnalysis.n_bs)


        * [`SurfaceEnergy`](maml.apps.pes.md#maml.apps.pes._lammps.SurfaceEnergy)


            * [`SurfaceEnergy.calculate()`](maml.apps.pes.md#maml.apps.pes._lammps.SurfaceEnergy.calculate)


        * [`_pretty_input()`](maml.apps.pes.md#maml.apps.pes._lammps._pretty_input)


        * [`_read_dump()`](maml.apps.pes.md#maml.apps.pes._lammps._read_dump)


        * [`get_default_lmp_exe()`](maml.apps.pes.md#maml.apps.pes._lammps.get_default_lmp_exe)


    * [maml.apps.pes._mtp module](maml.apps.pes.md#module-maml.apps.pes._mtp)


        * [`MTPotential`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential)


            * [`MTPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential._abc_impl)


            * [`MTPotential._line_up()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential._line_up)


            * [`MTPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.evaluate)


            * [`MTPotential.from_config()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.from_config)


            * [`MTPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.pair_coeff)


            * [`MTPotential.pair_style`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.pair_style)


            * [`MTPotential.read_cfgs()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.read_cfgs)


            * [`MTPotential.train()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.train)


            * [`MTPotential.write_cfg()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.write_cfg)


            * [`MTPotential.write_ini()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.write_ini)


            * [`MTPotential.write_param()`](maml.apps.pes.md#maml.apps.pes._mtp.MTPotential.write_param)


        * [`feed()`](maml.apps.pes.md#maml.apps.pes._mtp.feed)


    * [maml.apps.pes._nnp module](maml.apps.pes.md#module-maml.apps.pes._nnp)


        * [`NNPotential`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential)


            * [`NNPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential._abc_impl)


            * [`NNPotential._line_up()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential._line_up)


            * [`NNPotential.bohr_to_angstrom`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.bohr_to_angstrom)


            * [`NNPotential.eV_to_Ha`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.eV_to_Ha)


            * [`NNPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.evaluate)


            * [`NNPotential.from_config()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.from_config)


            * [`NNPotential.load_input()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.load_input)


            * [`NNPotential.load_scaler()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.load_scaler)


            * [`NNPotential.load_weights()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.load_weights)


            * [`NNPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.pair_coeff)


            * [`NNPotential.pair_style`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.pair_style)


            * [`NNPotential.read_cfgs()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.read_cfgs)


            * [`NNPotential.train()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.train)


            * [`NNPotential.write_cfgs()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.write_cfgs)


            * [`NNPotential.write_input()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.write_input)


            * [`NNPotential.write_param()`](maml.apps.pes.md#maml.apps.pes._nnp.NNPotential.write_param)


    * [maml.apps.pes._snap module](maml.apps.pes.md#module-maml.apps.pes._snap)


        * [`SNAPotential`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential)


            * [`SNAPotential._abc_impl`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential._abc_impl)


            * [`SNAPotential.evaluate()`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential.evaluate)


            * [`SNAPotential.from_config()`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential.from_config)


            * [`SNAPotential.pair_coeff`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential.pair_coeff)


            * [`SNAPotential.pair_style`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential.pair_style)


            * [`SNAPotential.train()`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential.train)


            * [`SNAPotential.write_param()`](maml.apps.pes.md#maml.apps.pes._snap.SNAPotential.write_param)


* [maml.apps.symbolic package](maml.apps.symbolic.md)


    * [`AdaptiveLasso`](maml.apps.symbolic.md#maml.apps.symbolic.AdaptiveLasso)


        * [`AdaptiveLasso._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic.AdaptiveLasso._penalty_jac)


        * [`AdaptiveLasso.get_w()`](maml.apps.symbolic.md#maml.apps.symbolic.AdaptiveLasso.get_w)


        * [`AdaptiveLasso.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic.AdaptiveLasso.penalty)


        * [`AdaptiveLasso.select()`](maml.apps.symbolic.md#maml.apps.symbolic.AdaptiveLasso.select)


    * [`DantzigSelector`](maml.apps.symbolic.md#maml.apps.symbolic.DantzigSelector)


        * [`DantzigSelector.construct_constraints()`](maml.apps.symbolic.md#maml.apps.symbolic.DantzigSelector.construct_constraints)


        * [`DantzigSelector.construct_jac()`](maml.apps.symbolic.md#maml.apps.symbolic.DantzigSelector.construct_jac)


        * [`DantzigSelector.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic.DantzigSelector.construct_loss)


    * [`FeatureGenerator`](maml.apps.symbolic.md#maml.apps.symbolic.FeatureGenerator)


        * [`FeatureGenerator.augment()`](maml.apps.symbolic.md#maml.apps.symbolic.FeatureGenerator.augment)


    * [`ISIS`](maml.apps.symbolic.md#maml.apps.symbolic.ISIS)


        * [`ISIS.evaluate()`](maml.apps.symbolic.md#maml.apps.symbolic.ISIS.evaluate)


        * [`ISIS.run()`](maml.apps.symbolic.md#maml.apps.symbolic.ISIS.run)


    * [`L0BrutalForce`](maml.apps.symbolic.md#maml.apps.symbolic.L0BrutalForce)


        * [`L0BrutalForce.select()`](maml.apps.symbolic.md#maml.apps.symbolic.L0BrutalForce.select)


    * [`Lasso`](maml.apps.symbolic.md#maml.apps.symbolic.Lasso)


        * [`Lasso._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic.Lasso._penalty_jac)


        * [`Lasso.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic.Lasso.penalty)


    * [`Operator`](maml.apps.symbolic.md#maml.apps.symbolic.Operator)


        * [`Operator.compute()`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.compute)


        * [`Operator.from_str()`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.from_str)


        * [`Operator.gen_name()`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.gen_name)


        * [`Operator.is_binary`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.is_binary)


        * [`Operator.is_commutative`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.is_commutative)


        * [`Operator.is_unary`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.is_unary)


        * [`Operator.support_op_rep`](maml.apps.symbolic.md#maml.apps.symbolic.Operator.support_op_rep)


    * [`SCAD`](maml.apps.symbolic.md#maml.apps.symbolic.SCAD)


        * [`SCAD._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic.SCAD._penalty_jac)


        * [`SCAD.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic.SCAD.penalty)


    * [`SIS`](maml.apps.symbolic.md#maml.apps.symbolic.SIS)


        * [`SIS.compute_residual()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.compute_residual)


        * [`SIS.run()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.run)


        * [`SIS.screen()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.screen)


        * [`SIS.select()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.select)


        * [`SIS.set_gamma()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.set_gamma)


        * [`SIS.set_selector()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.set_selector)


        * [`SIS.update_gamma()`](maml.apps.symbolic.md#maml.apps.symbolic.SIS.update_gamma)




    * [maml.apps.symbolic._feature_generator module](maml.apps.symbolic.md#module-maml.apps.symbolic._feature_generator)


        * [`FeatureGenerator`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.FeatureGenerator)


            * [`FeatureGenerator.augment()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.FeatureGenerator.augment)


        * [`Operator`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator)


            * [`Operator.compute()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.compute)


            * [`Operator.from_str()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.from_str)


            * [`Operator.gen_name()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.gen_name)


            * [`Operator.is_binary`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.is_binary)


            * [`Operator.is_commutative`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.is_commutative)


            * [`Operator.is_unary`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.is_unary)


            * [`Operator.support_op_rep`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.Operator.support_op_rep)


        * [`_my_abs_diff()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_abs_diff)


        * [`_my_abs_log10()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_abs_log10)


        * [`_my_abs_sqrt()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_abs_sqrt)


        * [`_my_abs_sum()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_abs_sum)


        * [`_my_diff()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_diff)


        * [`_my_div()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_div)


        * [`_my_exp()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_exp)


        * [`_my_exp_power_2()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_exp_power_2)


        * [`_my_exp_power_3()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_exp_power_3)


        * [`_my_mul()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_mul)


        * [`_my_power()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_power)


        * [`_my_sum()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_sum)


        * [`_my_sum_exp()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_sum_exp)


        * [`_my_sum_power_2()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_sum_power_2)


        * [`_my_sum_power_3()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._my_sum_power_3)


        * [`_update_df()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator._update_df)


        * [`generate_feature()`](maml.apps.symbolic.md#maml.apps.symbolic._feature_generator.generate_feature)


    * [maml.apps.symbolic._selectors module](maml.apps.symbolic.md#module-maml.apps.symbolic._selectors)


        * [`AdaptiveLasso`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.AdaptiveLasso)


            * [`AdaptiveLasso._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.AdaptiveLasso._penalty_jac)


            * [`AdaptiveLasso.get_w()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.AdaptiveLasso.get_w)


            * [`AdaptiveLasso.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.AdaptiveLasso.penalty)


            * [`AdaptiveLasso.select()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.AdaptiveLasso.select)


        * [`BaseSelector`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector)


            * [`BaseSelector._get_param_names()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector._get_param_names)


            * [`BaseSelector.compute_residual()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.compute_residual)


            * [`BaseSelector.construct_constraints()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.construct_constraints)


            * [`BaseSelector.construct_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.construct_jac)


            * [`BaseSelector.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.construct_loss)


            * [`BaseSelector.evaluate()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.evaluate)


            * [`BaseSelector.get_coef()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.get_coef)


            * [`BaseSelector.get_feature_indices()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.get_feature_indices)


            * [`BaseSelector.get_params()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.get_params)


            * [`BaseSelector.predict()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.predict)


            * [`BaseSelector.select()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.select)


            * [`BaseSelector.set_params()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.BaseSelector.set_params)


        * [`DantzigSelector`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.DantzigSelector)


            * [`DantzigSelector.construct_constraints()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.DantzigSelector.construct_constraints)


            * [`DantzigSelector.construct_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.DantzigSelector.construct_jac)


            * [`DantzigSelector.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.DantzigSelector.construct_loss)


        * [`L0BrutalForce`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.L0BrutalForce)


            * [`L0BrutalForce.select()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.L0BrutalForce.select)


        * [`Lasso`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.Lasso)


            * [`Lasso._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.Lasso._penalty_jac)


            * [`Lasso.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.Lasso.penalty)


        * [`PenalizedLeastSquares`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares)


            * [`PenalizedLeastSquares._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares._penalty_jac)


            * [`PenalizedLeastSquares._sse_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares._sse_jac)


            * [`PenalizedLeastSquares.construct_constraints()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares.construct_constraints)


            * [`PenalizedLeastSquares.construct_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares.construct_jac)


            * [`PenalizedLeastSquares.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares.construct_loss)


            * [`PenalizedLeastSquares.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.PenalizedLeastSquares.penalty)


        * [`SCAD`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.SCAD)


            * [`SCAD._penalty_jac()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.SCAD._penalty_jac)


            * [`SCAD.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors.SCAD.penalty)


    * [maml.apps.symbolic._selectors_cvxpy module](maml.apps.symbolic.md#module-maml.apps.symbolic._selectors_cvxpy)


        * [`AdaptiveLassoCP`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.AdaptiveLassoCP)


            * [`AdaptiveLassoCP.get_w()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.AdaptiveLassoCP.get_w)


            * [`AdaptiveLassoCP.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.AdaptiveLassoCP.penalty)


            * [`AdaptiveLassoCP.select()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.AdaptiveLassoCP.select)


        * [`BaseSelectorCP`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.BaseSelectorCP)


            * [`BaseSelectorCP.construct_constraints()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.BaseSelectorCP.construct_constraints)


            * [`BaseSelectorCP.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.BaseSelectorCP.construct_loss)


            * [`BaseSelectorCP.select()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.BaseSelectorCP.select)


        * [`DantzigSelectorCP`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.DantzigSelectorCP)


            * [`DantzigSelectorCP.construct_constraints()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.DantzigSelectorCP.construct_constraints)


            * [`DantzigSelectorCP.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.DantzigSelectorCP.construct_loss)


        * [`LassoCP`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.LassoCP)


            * [`LassoCP.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.LassoCP.penalty)


        * [`PenalizedLeastSquaresCP`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.PenalizedLeastSquaresCP)


            * [`PenalizedLeastSquaresCP.construct_loss()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.PenalizedLeastSquaresCP.construct_loss)


            * [`PenalizedLeastSquaresCP.penalty()`](maml.apps.symbolic.md#maml.apps.symbolic._selectors_cvxpy.PenalizedLeastSquaresCP.penalty)


    * [maml.apps.symbolic._sis module](maml.apps.symbolic.md#module-maml.apps.symbolic._sis)


        * [`ISIS`](maml.apps.symbolic.md#maml.apps.symbolic._sis.ISIS)


            * [`ISIS.evaluate()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.ISIS.evaluate)


            * [`ISIS.run()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.ISIS.run)


        * [`SIS`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS)


            * [`SIS.compute_residual()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.compute_residual)


            * [`SIS.run()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.run)


            * [`SIS.screen()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.screen)


            * [`SIS.select()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.select)


            * [`SIS.set_gamma()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.set_gamma)


            * [`SIS.set_selector()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.set_selector)


            * [`SIS.update_gamma()`](maml.apps.symbolic.md#maml.apps.symbolic._sis.SIS.update_gamma)


        * [`_best_combination()`](maml.apps.symbolic.md#maml.apps.symbolic._sis._best_combination)


        * [`_eval()`](maml.apps.symbolic.md#maml.apps.symbolic._sis._eval)


        * [`_get_coeff()`](maml.apps.symbolic.md#maml.apps.symbolic._sis._get_coeff)