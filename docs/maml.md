---
layout: default
title: API Documentation
nav_order: 5
---

# maml package

## Subpackages


* [maml.apps package](maml.apps.md)


    * [Subpackages](maml.apps.md#subpackages)


        * [maml.apps.bowsr package](maml.apps.bowsr.md)


            * [Subpackages](maml.apps.bowsr.md#subpackages)


                * [maml.apps.bowsr.model package](maml.apps.bowsr.model.md)


                    * [`EnergyModel`](maml.apps.bowsr.model.md#maml.apps.bowsr.model.EnergyModel)




                    * [maml.apps.bowsr.model.base module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.base)


                    * [maml.apps.bowsr.model.cgcnn module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.cgcnn)


                    * [maml.apps.bowsr.model.dft module](maml.apps.bowsr.model.md#module-maml.apps.bowsr.model.dft)


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


* [maml.base package](maml.base.md)


    * [`BaseDataSource`](maml.base.md#maml.base.BaseDataSource)


        * [`BaseDataSource.get()`](maml.base.md#maml.base.BaseDataSource.get)


    * [`BaseDescriber`](maml.base.md#maml.base.BaseDescriber)


        * [`BaseDescriber._abc_impl`](maml.base.md#maml.base.BaseDescriber._abc_impl)


        * [`BaseDescriber._is_multi_output()`](maml.base.md#maml.base.BaseDescriber._is_multi_output)


        * [`BaseDescriber._sklearn_auto_wrap_output_keys`](maml.base.md#maml.base.BaseDescriber._sklearn_auto_wrap_output_keys)


        * [`BaseDescriber.clear_cache()`](maml.base.md#maml.base.BaseDescriber.clear_cache)


        * [`BaseDescriber.feature_dim`](maml.base.md#maml.base.BaseDescriber.feature_dim)


        * [`BaseDescriber.fit()`](maml.base.md#maml.base.BaseDescriber.fit)


        * [`BaseDescriber.transform()`](maml.base.md#maml.base.BaseDescriber.transform)


        * [`BaseDescriber.transform_one()`](maml.base.md#maml.base.BaseDescriber.transform_one)


    * [`BaseModel`](maml.base.md#maml.base.BaseModel)


        * [`BaseModel._predict()`](maml.base.md#maml.base.BaseModel._predict)


        * [`BaseModel.fit()`](maml.base.md#maml.base.BaseModel.fit)


        * [`BaseModel.predict_objs()`](maml.base.md#maml.base.BaseModel.predict_objs)


        * [`BaseModel.train()`](maml.base.md#maml.base.BaseModel.train)


    * [`DummyDescriber`](maml.base.md#maml.base.DummyDescriber)


        * [`DummyDescriber._abc_impl`](maml.base.md#maml.base.DummyDescriber._abc_impl)


        * [`DummyDescriber._sklearn_auto_wrap_output_keys`](maml.base.md#maml.base.DummyDescriber._sklearn_auto_wrap_output_keys)


        * [`DummyDescriber.transform_one()`](maml.base.md#maml.base.DummyDescriber.transform_one)


    * [`KerasModel`](maml.base.md#maml.base.KerasModel)


        * [`KerasModel._get_validation_data()`](maml.base.md#maml.base.KerasModel._get_validation_data)


        * [`KerasModel.fit()`](maml.base.md#maml.base.KerasModel.fit)


    * [`SKLModel`](maml.base.md#maml.base.SKLModel)


    * [`SequentialDescriber`](maml.base.md#maml.base.SequentialDescriber)


        * [`SequentialDescriber._abc_impl`](maml.base.md#maml.base.SequentialDescriber._abc_impl)


        * [`SequentialDescriber.steps`](maml.base.md#maml.base.SequentialDescriber.steps)


    * [`TargetScalerMixin`](maml.base.md#maml.base.TargetScalerMixin)


        * [`TargetScalerMixin.predict_objs()`](maml.base.md#maml.base.TargetScalerMixin.predict_objs)


        * [`TargetScalerMixin.train()`](maml.base.md#maml.base.TargetScalerMixin.train)


    * [`describer_type()`](maml.base.md#maml.base.describer_type)


    * [`get_feature_batch()`](maml.base.md#maml.base.get_feature_batch)


    * [`is_keras_model()`](maml.base.md#maml.base.is_keras_model)


    * [`is_sklearn_model()`](maml.base.md#maml.base.is_sklearn_model)




    * [maml.base._data module](maml.base.md#module-maml.base._data)


        * [`BaseDataSource`](maml.base.md#maml.base._data.BaseDataSource)


            * [`BaseDataSource.get()`](maml.base.md#maml.base._data.BaseDataSource.get)


    * [maml.base._describer module](maml.base.md#module-maml.base._describer)


        * [`BaseDescriber`](maml.base.md#maml.base._describer.BaseDescriber)


            * [`BaseDescriber._abc_impl`](maml.base.md#maml.base._describer.BaseDescriber._abc_impl)


            * [`BaseDescriber._is_multi_output()`](maml.base.md#maml.base._describer.BaseDescriber._is_multi_output)


            * [`BaseDescriber._sklearn_auto_wrap_output_keys`](maml.base.md#maml.base._describer.BaseDescriber._sklearn_auto_wrap_output_keys)


            * [`BaseDescriber.clear_cache()`](maml.base.md#maml.base._describer.BaseDescriber.clear_cache)


            * [`BaseDescriber.feature_dim`](maml.base.md#maml.base._describer.BaseDescriber.feature_dim)


            * [`BaseDescriber.fit()`](maml.base.md#maml.base._describer.BaseDescriber.fit)


            * [`BaseDescriber.transform()`](maml.base.md#maml.base._describer.BaseDescriber.transform)


            * [`BaseDescriber.transform_one()`](maml.base.md#maml.base._describer.BaseDescriber.transform_one)


        * [`DummyDescriber`](maml.base.md#maml.base._describer.DummyDescriber)


            * [`DummyDescriber._abc_impl`](maml.base.md#maml.base._describer.DummyDescriber._abc_impl)


            * [`DummyDescriber._sklearn_auto_wrap_output_keys`](maml.base.md#maml.base._describer.DummyDescriber._sklearn_auto_wrap_output_keys)


            * [`DummyDescriber.transform_one()`](maml.base.md#maml.base._describer.DummyDescriber.transform_one)


        * [`SequentialDescriber`](maml.base.md#maml.base._describer.SequentialDescriber)


            * [`SequentialDescriber._abc_impl`](maml.base.md#maml.base._describer.SequentialDescriber._abc_impl)


            * [`SequentialDescriber.steps`](maml.base.md#maml.base._describer.SequentialDescriber.steps)


        * [`_transform_one()`](maml.base.md#maml.base._describer._transform_one)


        * [`describer_type()`](maml.base.md#maml.base._describer.describer_type)


    * [maml.base._feature_batch module](maml.base.md#module-maml.base._feature_batch)


        * [`get_feature_batch()`](maml.base.md#maml.base._feature_batch.get_feature_batch)


        * [`no_action()`](maml.base.md#maml.base._feature_batch.no_action)


        * [`pandas_concat()`](maml.base.md#maml.base._feature_batch.pandas_concat)


        * [`stack_first_dim()`](maml.base.md#maml.base._feature_batch.stack_first_dim)


        * [`stack_padded()`](maml.base.md#maml.base._feature_batch.stack_padded)


    * [maml.base._mixin module](maml.base.md#module-maml.base._mixin)


        * [`TargetScalerMixin`](maml.base.md#maml.base._mixin.TargetScalerMixin)


            * [`TargetScalerMixin.predict_objs()`](maml.base.md#maml.base._mixin.TargetScalerMixin.predict_objs)


            * [`TargetScalerMixin.train()`](maml.base.md#maml.base._mixin.TargetScalerMixin.train)


    * [maml.base._model module](maml.base.md#module-maml.base._model)


        * [`BaseModel`](maml.base.md#maml.base._model.BaseModel)


            * [`BaseModel._predict()`](maml.base.md#maml.base._model.BaseModel._predict)


            * [`BaseModel.fit()`](maml.base.md#maml.base._model.BaseModel.fit)


            * [`BaseModel.predict_objs()`](maml.base.md#maml.base._model.BaseModel.predict_objs)


            * [`BaseModel.train()`](maml.base.md#maml.base._model.BaseModel.train)


        * [`KerasMixin`](maml.base.md#maml.base._model.KerasMixin)


            * [`KerasMixin.evaluate()`](maml.base.md#maml.base._model.KerasMixin.evaluate)


            * [`KerasMixin.from_file()`](maml.base.md#maml.base._model.KerasMixin.from_file)


            * [`KerasMixin.get_input_dim()`](maml.base.md#maml.base._model.KerasMixin.get_input_dim)


            * [`KerasMixin.load()`](maml.base.md#maml.base._model.KerasMixin.load)


            * [`KerasMixin.save()`](maml.base.md#maml.base._model.KerasMixin.save)


        * [`KerasModel`](maml.base.md#maml.base._model.KerasModel)


            * [`KerasModel._get_validation_data()`](maml.base.md#maml.base._model.KerasModel._get_validation_data)


            * [`KerasModel.fit()`](maml.base.md#maml.base._model.KerasModel.fit)


        * [`SKLModel`](maml.base.md#maml.base._model.SKLModel)


        * [`SklearnMixin`](maml.base.md#maml.base._model.SklearnMixin)


            * [`SklearnMixin.evaluate()`](maml.base.md#maml.base._model.SklearnMixin.evaluate)


            * [`SklearnMixin.from_file()`](maml.base.md#maml.base._model.SklearnMixin.from_file)


            * [`SklearnMixin.load()`](maml.base.md#maml.base._model.SklearnMixin.load)


            * [`SklearnMixin.save()`](maml.base.md#maml.base._model.SklearnMixin.save)


        * [`is_keras_model()`](maml.base.md#maml.base._model.is_keras_model)


        * [`is_sklearn_model()`](maml.base.md#maml.base._model.is_sklearn_model)


* [maml.data package](maml.data.md)


    * [`MaterialsProject`](maml.data.md#maml.data.MaterialsProject)


        * [`MaterialsProject.get()`](maml.data.md#maml.data.MaterialsProject.get)


    * [`URLSource`](maml.data.md#maml.data.URLSource)


        * [`URLSource.get()`](maml.data.md#maml.data.URLSource.get)




    * [maml.data._mp module](maml.data.md#module-maml.data._mp)


        * [`MaterialsProject`](maml.data.md#maml.data._mp.MaterialsProject)


            * [`MaterialsProject.get()`](maml.data.md#maml.data._mp.MaterialsProject.get)


    * [maml.data._url module](maml.data.md#module-maml.data._url)


        * [`FigshareSource`](maml.data.md#maml.data._url.FigshareSource)


            * [`FigshareSource.get()`](maml.data.md#maml.data._url.FigshareSource.get)


        * [`URLSource`](maml.data.md#maml.data._url.URLSource)


            * [`URLSource.get()`](maml.data.md#maml.data._url.URLSource.get)


* [maml.describers package](maml.describers.md)


    * [`BPSymmetryFunctions`](maml.describers.md#maml.describers.BPSymmetryFunctions)


        * [`BPSymmetryFunctions._abc_impl`](maml.describers.md#maml.describers.BPSymmetryFunctions._abc_impl)


        * [`BPSymmetryFunctions._fc()`](maml.describers.md#maml.describers.BPSymmetryFunctions._fc)


        * [`BPSymmetryFunctions._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.BPSymmetryFunctions._sklearn_auto_wrap_output_keys)


        * [`BPSymmetryFunctions.describer_type`](maml.describers.md#maml.describers.BPSymmetryFunctions.describer_type)


        * [`BPSymmetryFunctions.transform_one()`](maml.describers.md#maml.describers.BPSymmetryFunctions.transform_one)


    * [`BispectrumCoefficients`](maml.describers.md#maml.describers.BispectrumCoefficients)


        * [`BispectrumCoefficients._abc_impl`](maml.describers.md#maml.describers.BispectrumCoefficients._abc_impl)


        * [`BispectrumCoefficients._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.BispectrumCoefficients._sklearn_auto_wrap_output_keys)


        * [`BispectrumCoefficients.describer_type`](maml.describers.md#maml.describers.BispectrumCoefficients.describer_type)


        * [`BispectrumCoefficients.feature_dim`](maml.describers.md#maml.describers.BispectrumCoefficients.feature_dim)


        * [`BispectrumCoefficients.subscripts`](maml.describers.md#maml.describers.BispectrumCoefficients.subscripts)


        * [`BispectrumCoefficients.transform_one()`](maml.describers.md#maml.describers.BispectrumCoefficients.transform_one)


    * [`CoulombEigenSpectrum`](maml.describers.md#maml.describers.CoulombEigenSpectrum)


        * [`CoulombEigenSpectrum._abc_impl`](maml.describers.md#maml.describers.CoulombEigenSpectrum._abc_impl)


        * [`CoulombEigenSpectrum._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.CoulombEigenSpectrum._sklearn_auto_wrap_output_keys)


        * [`CoulombEigenSpectrum.describer_type`](maml.describers.md#maml.describers.CoulombEigenSpectrum.describer_type)


        * [`CoulombEigenSpectrum.transform_one()`](maml.describers.md#maml.describers.CoulombEigenSpectrum.transform_one)


    * [`CoulombMatrix`](maml.describers.md#maml.describers.CoulombMatrix)


        * [`CoulombMatrix._abc_impl`](maml.describers.md#maml.describers.CoulombMatrix._abc_impl)


        * [`CoulombMatrix._get_columb_mat()`](maml.describers.md#maml.describers.CoulombMatrix._get_columb_mat)


        * [`CoulombMatrix._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.CoulombMatrix._sklearn_auto_wrap_output_keys)


        * [`CoulombMatrix.describer_type`](maml.describers.md#maml.describers.CoulombMatrix.describer_type)


        * [`CoulombMatrix.get_coulomb_mat()`](maml.describers.md#maml.describers.CoulombMatrix.get_coulomb_mat)


        * [`CoulombMatrix.transform_one()`](maml.describers.md#maml.describers.CoulombMatrix.transform_one)


    * [`DistinctSiteProperty`](maml.describers.md#maml.describers.DistinctSiteProperty)


        * [`DistinctSiteProperty._abc_impl`](maml.describers.md#maml.describers.DistinctSiteProperty._abc_impl)


        * [`DistinctSiteProperty._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.DistinctSiteProperty._sklearn_auto_wrap_output_keys)


        * [`DistinctSiteProperty.describer_type`](maml.describers.md#maml.describers.DistinctSiteProperty.describer_type)


        * [`DistinctSiteProperty.supported_properties`](maml.describers.md#maml.describers.DistinctSiteProperty.supported_properties)


        * [`DistinctSiteProperty.transform_one()`](maml.describers.md#maml.describers.DistinctSiteProperty.transform_one)


    * [`ElementProperty`](maml.describers.md#maml.describers.ElementProperty)


        * [`ElementProperty._abc_impl`](maml.describers.md#maml.describers.ElementProperty._abc_impl)


        * [`ElementProperty._get_param_names()`](maml.describers.md#maml.describers.ElementProperty._get_param_names)


        * [`ElementProperty._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.ElementProperty._sklearn_auto_wrap_output_keys)


        * [`ElementProperty.describer_type`](maml.describers.md#maml.describers.ElementProperty.describer_type)


        * [`ElementProperty.from_preset()`](maml.describers.md#maml.describers.ElementProperty.from_preset)


        * [`ElementProperty.get_params()`](maml.describers.md#maml.describers.ElementProperty.get_params)


        * [`ElementProperty.transform_one()`](maml.describers.md#maml.describers.ElementProperty.transform_one)


    * [`ElementStats`](maml.describers.md#maml.describers.ElementStats)


        * [`ElementStats.ALLOWED_STATS`](maml.describers.md#maml.describers.ElementStats.ALLOWED_STATS)


        * [`ElementStats.AVAILABLE_DATA`](maml.describers.md#maml.describers.ElementStats.AVAILABLE_DATA)


        * [`ElementStats._abc_impl`](maml.describers.md#maml.describers.ElementStats._abc_impl)


        * [`ElementStats._reduce_dimension()`](maml.describers.md#maml.describers.ElementStats._reduce_dimension)


        * [`ElementStats._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.ElementStats._sklearn_auto_wrap_output_keys)


        * [`ElementStats.describer_type`](maml.describers.md#maml.describers.ElementStats.describer_type)


        * [`ElementStats.from_data()`](maml.describers.md#maml.describers.ElementStats.from_data)


        * [`ElementStats.from_file()`](maml.describers.md#maml.describers.ElementStats.from_file)


        * [`ElementStats.transform_one()`](maml.describers.md#maml.describers.ElementStats.transform_one)


    * [`M3GNetStructure`](maml.describers.md#maml.describers.M3GNetStructure)


        * [`M3GNetStructure._abc_impl`](maml.describers.md#maml.describers.M3GNetStructure._abc_impl)


        * [`M3GNetStructure._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.M3GNetStructure._sklearn_auto_wrap_output_keys)


        * [`M3GNetStructure.transform_one()`](maml.describers.md#maml.describers.M3GNetStructure.transform_one)


    * [`MEGNetSite`](maml.describers.md#maml.describers.MEGNetSite)


        * [`MEGNetSite._abc_impl`](maml.describers.md#maml.describers.MEGNetSite._abc_impl)


        * [`MEGNetSite._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.MEGNetSite._sklearn_auto_wrap_output_keys)


        * [`MEGNetSite.describer_type`](maml.describers.md#maml.describers.MEGNetSite.describer_type)


        * [`MEGNetSite.transform_one()`](maml.describers.md#maml.describers.MEGNetSite.transform_one)


    * [`MEGNetStructure`](maml.describers.md#maml.describers.MEGNetStructure)


        * [`MEGNetStructure._abc_impl`](maml.describers.md#maml.describers.MEGNetStructure._abc_impl)


        * [`MEGNetStructure._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.MEGNetStructure._sklearn_auto_wrap_output_keys)


        * [`MEGNetStructure.describer_type`](maml.describers.md#maml.describers.MEGNetStructure.describer_type)


        * [`MEGNetStructure.transform_one()`](maml.describers.md#maml.describers.MEGNetStructure.transform_one)


    * [`RadialDistributionFunction`](maml.describers.md#maml.describers.RadialDistributionFunction)


        * [`RadialDistributionFunction._get_specie_density()`](maml.describers.md#maml.describers.RadialDistributionFunction._get_specie_density)


        * [`RadialDistributionFunction.get_site_coordination()`](maml.describers.md#maml.describers.RadialDistributionFunction.get_site_coordination)


        * [`RadialDistributionFunction.get_site_rdf()`](maml.describers.md#maml.describers.RadialDistributionFunction.get_site_rdf)


        * [`RadialDistributionFunction.get_species_coordination()`](maml.describers.md#maml.describers.RadialDistributionFunction.get_species_coordination)


        * [`RadialDistributionFunction.get_species_rdf()`](maml.describers.md#maml.describers.RadialDistributionFunction.get_species_rdf)


    * [`RandomizedCoulombMatrix`](maml.describers.md#maml.describers.RandomizedCoulombMatrix)


        * [`RandomizedCoulombMatrix._abc_impl`](maml.describers.md#maml.describers.RandomizedCoulombMatrix._abc_impl)


        * [`RandomizedCoulombMatrix._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.RandomizedCoulombMatrix._sklearn_auto_wrap_output_keys)


        * [`RandomizedCoulombMatrix.describer_type`](maml.describers.md#maml.describers.RandomizedCoulombMatrix.describer_type)


        * [`RandomizedCoulombMatrix.get_randomized_coulomb_mat()`](maml.describers.md#maml.describers.RandomizedCoulombMatrix.get_randomized_coulomb_mat)


        * [`RandomizedCoulombMatrix.transform_one()`](maml.describers.md#maml.describers.RandomizedCoulombMatrix.transform_one)


    * [`SiteElementProperty`](maml.describers.md#maml.describers.SiteElementProperty)


        * [`SiteElementProperty._abc_impl`](maml.describers.md#maml.describers.SiteElementProperty._abc_impl)


        * [`SiteElementProperty._get_keys()`](maml.describers.md#maml.describers.SiteElementProperty._get_keys)


        * [`SiteElementProperty._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.SiteElementProperty._sklearn_auto_wrap_output_keys)


        * [`SiteElementProperty.describer_type`](maml.describers.md#maml.describers.SiteElementProperty.describer_type)


        * [`SiteElementProperty.feature_dim`](maml.describers.md#maml.describers.SiteElementProperty.feature_dim)


        * [`SiteElementProperty.transform_one()`](maml.describers.md#maml.describers.SiteElementProperty.transform_one)


    * [`SmoothOverlapAtomicPosition`](maml.describers.md#maml.describers.SmoothOverlapAtomicPosition)


        * [`SmoothOverlapAtomicPosition._abc_impl`](maml.describers.md#maml.describers.SmoothOverlapAtomicPosition._abc_impl)


        * [`SmoothOverlapAtomicPosition._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.SmoothOverlapAtomicPosition._sklearn_auto_wrap_output_keys)


        * [`SmoothOverlapAtomicPosition.describer_type`](maml.describers.md#maml.describers.SmoothOverlapAtomicPosition.describer_type)


        * [`SmoothOverlapAtomicPosition.transform_one()`](maml.describers.md#maml.describers.SmoothOverlapAtomicPosition.transform_one)


    * [`SortedCoulombMatrix`](maml.describers.md#maml.describers.SortedCoulombMatrix)


        * [`SortedCoulombMatrix._abc_impl`](maml.describers.md#maml.describers.SortedCoulombMatrix._abc_impl)


        * [`SortedCoulombMatrix._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers.SortedCoulombMatrix._sklearn_auto_wrap_output_keys)


        * [`SortedCoulombMatrix.describer_type`](maml.describers.md#maml.describers.SortedCoulombMatrix.describer_type)


        * [`SortedCoulombMatrix.get_sorted_coulomb_mat()`](maml.describers.md#maml.describers.SortedCoulombMatrix.get_sorted_coulomb_mat)


        * [`SortedCoulombMatrix.transform_one()`](maml.describers.md#maml.describers.SortedCoulombMatrix.transform_one)


    * [`wrap_matminer_describer()`](maml.describers.md#maml.describers.wrap_matminer_describer)




    * [maml.describers._composition module](maml.describers.md#module-maml.describers._composition)


        * [`ElementStats`](maml.describers.md#maml.describers._composition.ElementStats)


            * [`ElementStats.ALLOWED_STATS`](maml.describers.md#maml.describers._composition.ElementStats.ALLOWED_STATS)


            * [`ElementStats.AVAILABLE_DATA`](maml.describers.md#maml.describers._composition.ElementStats.AVAILABLE_DATA)


            * [`ElementStats._abc_impl`](maml.describers.md#maml.describers._composition.ElementStats._abc_impl)


            * [`ElementStats._reduce_dimension()`](maml.describers.md#maml.describers._composition.ElementStats._reduce_dimension)


            * [`ElementStats._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._composition.ElementStats._sklearn_auto_wrap_output_keys)


            * [`ElementStats.describer_type`](maml.describers.md#maml.describers._composition.ElementStats.describer_type)


            * [`ElementStats.from_data()`](maml.describers.md#maml.describers._composition.ElementStats.from_data)


            * [`ElementStats.from_file()`](maml.describers.md#maml.describers._composition.ElementStats.from_file)


            * [`ElementStats.transform_one()`](maml.describers.md#maml.describers._composition.ElementStats.transform_one)


        * [`_is_element_or_specie()`](maml.describers.md#maml.describers._composition._is_element_or_specie)


        * [`_keys_are_elements()`](maml.describers.md#maml.describers._composition._keys_are_elements)


    * [maml.describers._m3gnet module](maml.describers.md#module-maml.describers._m3gnet)


        * [`M3GNetStructure`](maml.describers.md#maml.describers._m3gnet.M3GNetStructure)


            * [`M3GNetStructure._abc_impl`](maml.describers.md#maml.describers._m3gnet.M3GNetStructure._abc_impl)


            * [`M3GNetStructure._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._m3gnet.M3GNetStructure._sklearn_auto_wrap_output_keys)


            * [`M3GNetStructure.transform_one()`](maml.describers.md#maml.describers._m3gnet.M3GNetStructure.transform_one)


    * [maml.describers._matminer module](maml.describers.md#module-maml.describers._matminer)


        * [`wrap_matminer_describer()`](maml.describers.md#maml.describers._matminer.wrap_matminer_describer)


    * [maml.describers._megnet module](maml.describers.md#module-maml.describers._megnet)


        * [`MEGNetNotFound`](maml.describers.md#maml.describers._megnet.MEGNetNotFound)


        * [`MEGNetSite`](maml.describers.md#maml.describers._megnet.MEGNetSite)


            * [`MEGNetSite._abc_impl`](maml.describers.md#maml.describers._megnet.MEGNetSite._abc_impl)


            * [`MEGNetSite._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._megnet.MEGNetSite._sklearn_auto_wrap_output_keys)


            * [`MEGNetSite.describer_type`](maml.describers.md#maml.describers._megnet.MEGNetSite.describer_type)


            * [`MEGNetSite.transform_one()`](maml.describers.md#maml.describers._megnet.MEGNetSite.transform_one)


        * [`MEGNetStructure`](maml.describers.md#maml.describers._megnet.MEGNetStructure)


            * [`MEGNetStructure._abc_impl`](maml.describers.md#maml.describers._megnet.MEGNetStructure._abc_impl)


            * [`MEGNetStructure._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._megnet.MEGNetStructure._sklearn_auto_wrap_output_keys)


            * [`MEGNetStructure.describer_type`](maml.describers.md#maml.describers._megnet.MEGNetStructure.describer_type)


            * [`MEGNetStructure.transform_one()`](maml.describers.md#maml.describers._megnet.MEGNetStructure.transform_one)


        * [`_load_model()`](maml.describers.md#maml.describers._megnet._load_model)


    * [maml.describers._rdf module](maml.describers.md#module-maml.describers._rdf)


        * [`RadialDistributionFunction`](maml.describers.md#maml.describers._rdf.RadialDistributionFunction)


            * [`RadialDistributionFunction._get_specie_density()`](maml.describers.md#maml.describers._rdf.RadialDistributionFunction._get_specie_density)


            * [`RadialDistributionFunction.get_site_coordination()`](maml.describers.md#maml.describers._rdf.RadialDistributionFunction.get_site_coordination)


            * [`RadialDistributionFunction.get_site_rdf()`](maml.describers.md#maml.describers._rdf.RadialDistributionFunction.get_site_rdf)


            * [`RadialDistributionFunction.get_species_coordination()`](maml.describers.md#maml.describers._rdf.RadialDistributionFunction.get_species_coordination)


            * [`RadialDistributionFunction.get_species_rdf()`](maml.describers.md#maml.describers._rdf.RadialDistributionFunction.get_species_rdf)


        * [`_dist_to_counts()`](maml.describers.md#maml.describers._rdf._dist_to_counts)


        * [`get_pair_distances()`](maml.describers.md#maml.describers._rdf.get_pair_distances)


    * [maml.describers._site module](maml.describers.md#module-maml.describers._site)


        * [`BPSymmetryFunctions`](maml.describers.md#maml.describers._site.BPSymmetryFunctions)


            * [`BPSymmetryFunctions._abc_impl`](maml.describers.md#maml.describers._site.BPSymmetryFunctions._abc_impl)


            * [`BPSymmetryFunctions._fc()`](maml.describers.md#maml.describers._site.BPSymmetryFunctions._fc)


            * [`BPSymmetryFunctions._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._site.BPSymmetryFunctions._sklearn_auto_wrap_output_keys)


            * [`BPSymmetryFunctions.describer_type`](maml.describers.md#maml.describers._site.BPSymmetryFunctions.describer_type)


            * [`BPSymmetryFunctions.transform_one()`](maml.describers.md#maml.describers._site.BPSymmetryFunctions.transform_one)


        * [`BispectrumCoefficients`](maml.describers.md#maml.describers._site.BispectrumCoefficients)


            * [`BispectrumCoefficients._abc_impl`](maml.describers.md#maml.describers._site.BispectrumCoefficients._abc_impl)


            * [`BispectrumCoefficients._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._site.BispectrumCoefficients._sklearn_auto_wrap_output_keys)


            * [`BispectrumCoefficients.describer_type`](maml.describers.md#maml.describers._site.BispectrumCoefficients.describer_type)


            * [`BispectrumCoefficients.feature_dim`](maml.describers.md#maml.describers._site.BispectrumCoefficients.feature_dim)


            * [`BispectrumCoefficients.subscripts`](maml.describers.md#maml.describers._site.BispectrumCoefficients.subscripts)


            * [`BispectrumCoefficients.transform_one()`](maml.describers.md#maml.describers._site.BispectrumCoefficients.transform_one)


        * [`MEGNetSite`](maml.describers.md#maml.describers._site.MEGNetSite)


            * [`MEGNetSite._abc_impl`](maml.describers.md#maml.describers._site.MEGNetSite._abc_impl)


            * [`MEGNetSite._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._site.MEGNetSite._sklearn_auto_wrap_output_keys)


            * [`MEGNetSite.describer_type`](maml.describers.md#maml.describers._site.MEGNetSite.describer_type)


            * [`MEGNetSite.transform_one()`](maml.describers.md#maml.describers._site.MEGNetSite.transform_one)


        * [`SiteElementProperty`](maml.describers.md#maml.describers._site.SiteElementProperty)


            * [`SiteElementProperty._abc_impl`](maml.describers.md#maml.describers._site.SiteElementProperty._abc_impl)


            * [`SiteElementProperty._get_keys()`](maml.describers.md#maml.describers._site.SiteElementProperty._get_keys)


            * [`SiteElementProperty._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._site.SiteElementProperty._sklearn_auto_wrap_output_keys)


            * [`SiteElementProperty.describer_type`](maml.describers.md#maml.describers._site.SiteElementProperty.describer_type)


            * [`SiteElementProperty.feature_dim`](maml.describers.md#maml.describers._site.SiteElementProperty.feature_dim)


            * [`SiteElementProperty.transform_one()`](maml.describers.md#maml.describers._site.SiteElementProperty.transform_one)


        * [`SmoothOverlapAtomicPosition`](maml.describers.md#maml.describers._site.SmoothOverlapAtomicPosition)


            * [`SmoothOverlapAtomicPosition._abc_impl`](maml.describers.md#maml.describers._site.SmoothOverlapAtomicPosition._abc_impl)


            * [`SmoothOverlapAtomicPosition._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._site.SmoothOverlapAtomicPosition._sklearn_auto_wrap_output_keys)


            * [`SmoothOverlapAtomicPosition.describer_type`](maml.describers.md#maml.describers._site.SmoothOverlapAtomicPosition.describer_type)


            * [`SmoothOverlapAtomicPosition.transform_one()`](maml.describers.md#maml.describers._site.SmoothOverlapAtomicPosition.transform_one)


    * [maml.describers._spectrum module](maml.describers.md#module-maml.describers._spectrum)


    * [maml.describers._structure module](maml.describers.md#module-maml.describers._structure)


        * [`CoulombEigenSpectrum`](maml.describers.md#maml.describers._structure.CoulombEigenSpectrum)


            * [`CoulombEigenSpectrum._abc_impl`](maml.describers.md#maml.describers._structure.CoulombEigenSpectrum._abc_impl)


            * [`CoulombEigenSpectrum._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._structure.CoulombEigenSpectrum._sklearn_auto_wrap_output_keys)


            * [`CoulombEigenSpectrum.describer_type`](maml.describers.md#maml.describers._structure.CoulombEigenSpectrum.describer_type)


            * [`CoulombEigenSpectrum.transform_one()`](maml.describers.md#maml.describers._structure.CoulombEigenSpectrum.transform_one)


        * [`CoulombMatrix`](maml.describers.md#maml.describers._structure.CoulombMatrix)


            * [`CoulombMatrix._abc_impl`](maml.describers.md#maml.describers._structure.CoulombMatrix._abc_impl)


            * [`CoulombMatrix._get_columb_mat()`](maml.describers.md#maml.describers._structure.CoulombMatrix._get_columb_mat)


            * [`CoulombMatrix._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._structure.CoulombMatrix._sklearn_auto_wrap_output_keys)


            * [`CoulombMatrix.describer_type`](maml.describers.md#maml.describers._structure.CoulombMatrix.describer_type)


            * [`CoulombMatrix.get_coulomb_mat()`](maml.describers.md#maml.describers._structure.CoulombMatrix.get_coulomb_mat)


            * [`CoulombMatrix.transform_one()`](maml.describers.md#maml.describers._structure.CoulombMatrix.transform_one)


        * [`DistinctSiteProperty`](maml.describers.md#maml.describers._structure.DistinctSiteProperty)


            * [`DistinctSiteProperty._abc_impl`](maml.describers.md#maml.describers._structure.DistinctSiteProperty._abc_impl)


            * [`DistinctSiteProperty._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._structure.DistinctSiteProperty._sklearn_auto_wrap_output_keys)


            * [`DistinctSiteProperty.describer_type`](maml.describers.md#maml.describers._structure.DistinctSiteProperty.describer_type)


            * [`DistinctSiteProperty.supported_properties`](maml.describers.md#maml.describers._structure.DistinctSiteProperty.supported_properties)


            * [`DistinctSiteProperty.transform_one()`](maml.describers.md#maml.describers._structure.DistinctSiteProperty.transform_one)


        * [`RandomizedCoulombMatrix`](maml.describers.md#maml.describers._structure.RandomizedCoulombMatrix)


            * [`RandomizedCoulombMatrix._abc_impl`](maml.describers.md#maml.describers._structure.RandomizedCoulombMatrix._abc_impl)


            * [`RandomizedCoulombMatrix._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._structure.RandomizedCoulombMatrix._sklearn_auto_wrap_output_keys)


            * [`RandomizedCoulombMatrix.describer_type`](maml.describers.md#maml.describers._structure.RandomizedCoulombMatrix.describer_type)


            * [`RandomizedCoulombMatrix.get_randomized_coulomb_mat()`](maml.describers.md#maml.describers._structure.RandomizedCoulombMatrix.get_randomized_coulomb_mat)


            * [`RandomizedCoulombMatrix.transform_one()`](maml.describers.md#maml.describers._structure.RandomizedCoulombMatrix.transform_one)


        * [`SortedCoulombMatrix`](maml.describers.md#maml.describers._structure.SortedCoulombMatrix)


            * [`SortedCoulombMatrix._abc_impl`](maml.describers.md#maml.describers._structure.SortedCoulombMatrix._abc_impl)


            * [`SortedCoulombMatrix._sklearn_auto_wrap_output_keys`](maml.describers.md#maml.describers._structure.SortedCoulombMatrix._sklearn_auto_wrap_output_keys)


            * [`SortedCoulombMatrix.describer_type`](maml.describers.md#maml.describers._structure.SortedCoulombMatrix.describer_type)


            * [`SortedCoulombMatrix.get_sorted_coulomb_mat()`](maml.describers.md#maml.describers._structure.SortedCoulombMatrix.get_sorted_coulomb_mat)


            * [`SortedCoulombMatrix.transform_one()`](maml.describers.md#maml.describers._structure.SortedCoulombMatrix.transform_one)


* [maml.models package](maml.models.md)


    * [`AtomSets`](maml.models.md#maml.models.AtomSets)


        * [`AtomSets._get_data_generator()`](maml.models.md#maml.models.AtomSets._get_data_generator)


        * [`AtomSets._predict()`](maml.models.md#maml.models.AtomSets._predict)


        * [`AtomSets.evaluate()`](maml.models.md#maml.models.AtomSets.evaluate)


        * [`AtomSets.fit()`](maml.models.md#maml.models.AtomSets.fit)


        * [`AtomSets.from_dir()`](maml.models.md#maml.models.AtomSets.from_dir)


        * [`AtomSets.save()`](maml.models.md#maml.models.AtomSets.save)


    * [`KerasModel`](maml.models.md#maml.models.KerasModel)


        * [`KerasModel._get_validation_data()`](maml.models.md#maml.models.KerasModel._get_validation_data)


        * [`KerasModel.fit()`](maml.models.md#maml.models.KerasModel.fit)


    * [`MLP`](maml.models.md#maml.models.MLP)


    * [`SKLModel`](maml.models.md#maml.models.SKLModel)


    * [`WeightedAverageLayer`](maml.models.md#maml.models.WeightedAverageLayer)


        * [`WeightedAverageLayer.build()`](maml.models.md#maml.models.WeightedAverageLayer.build)


        * [`WeightedAverageLayer.call()`](maml.models.md#maml.models.WeightedAverageLayer.call)


        * [`WeightedAverageLayer.compute_output_shape()`](maml.models.md#maml.models.WeightedAverageLayer.compute_output_shape)


        * [`WeightedAverageLayer.get_config()`](maml.models.md#maml.models.WeightedAverageLayer.get_config)


        * [`WeightedAverageLayer.reduce_sum()`](maml.models.md#maml.models.WeightedAverageLayer.reduce_sum)


    * [`WeightedSet2Set`](maml.models.md#maml.models.WeightedSet2Set)


        * [`WeightedSet2Set._lstm()`](maml.models.md#maml.models.WeightedSet2Set._lstm)


        * [`WeightedSet2Set.build()`](maml.models.md#maml.models.WeightedSet2Set.build)


        * [`WeightedSet2Set.call()`](maml.models.md#maml.models.WeightedSet2Set.call)


        * [`WeightedSet2Set.compute_output_shape()`](maml.models.md#maml.models.WeightedSet2Set.compute_output_shape)


        * [`WeightedSet2Set.get_config()`](maml.models.md#maml.models.WeightedSet2Set.get_config)


    * [Subpackages](maml.models.md#subpackages)


        * [maml.models.dl package](maml.models.dl.md)


            * [`AtomSets`](maml.models.dl.md#maml.models.dl.AtomSets)


                * [`AtomSets._get_data_generator()`](maml.models.dl.md#maml.models.dl.AtomSets._get_data_generator)


                * [`AtomSets._predict()`](maml.models.dl.md#maml.models.dl.AtomSets._predict)


                * [`AtomSets.evaluate()`](maml.models.dl.md#maml.models.dl.AtomSets.evaluate)


                * [`AtomSets.fit()`](maml.models.dl.md#maml.models.dl.AtomSets.fit)


                * [`AtomSets.from_dir()`](maml.models.dl.md#maml.models.dl.AtomSets.from_dir)


                * [`AtomSets.save()`](maml.models.dl.md#maml.models.dl.AtomSets.save)


            * [`MLP`](maml.models.dl.md#maml.models.dl.MLP)


            * [`WeightedAverageLayer`](maml.models.dl.md#maml.models.dl.WeightedAverageLayer)


                * [`WeightedAverageLayer.build()`](maml.models.dl.md#maml.models.dl.WeightedAverageLayer.build)


                * [`WeightedAverageLayer.call()`](maml.models.dl.md#maml.models.dl.WeightedAverageLayer.call)


                * [`WeightedAverageLayer.compute_output_shape()`](maml.models.dl.md#maml.models.dl.WeightedAverageLayer.compute_output_shape)


                * [`WeightedAverageLayer.get_config()`](maml.models.dl.md#maml.models.dl.WeightedAverageLayer.get_config)


                * [`WeightedAverageLayer.reduce_sum()`](maml.models.dl.md#maml.models.dl.WeightedAverageLayer.reduce_sum)


            * [`WeightedSet2Set`](maml.models.dl.md#maml.models.dl.WeightedSet2Set)


                * [`WeightedSet2Set._lstm()`](maml.models.dl.md#maml.models.dl.WeightedSet2Set._lstm)


                * [`WeightedSet2Set.build()`](maml.models.dl.md#maml.models.dl.WeightedSet2Set.build)


                * [`WeightedSet2Set.call()`](maml.models.dl.md#maml.models.dl.WeightedSet2Set.call)


                * [`WeightedSet2Set.compute_output_shape()`](maml.models.dl.md#maml.models.dl.WeightedSet2Set.compute_output_shape)


                * [`WeightedSet2Set.get_config()`](maml.models.dl.md#maml.models.dl.WeightedSet2Set.get_config)




            * [maml.models.dl._atomsets module](maml.models.dl.md#module-maml.models.dl._atomsets)


                * [`AtomSets`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets)


                    * [`AtomSets._get_data_generator()`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets._get_data_generator)


                    * [`AtomSets._predict()`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets._predict)


                    * [`AtomSets.evaluate()`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets.evaluate)


                    * [`AtomSets.fit()`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets.fit)


                    * [`AtomSets.from_dir()`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets.from_dir)


                    * [`AtomSets.save()`](maml.models.dl.md#maml.models.dl._atomsets.AtomSets.save)


                * [`construct_atom_sets()`](maml.models.dl.md#maml.models.dl._atomsets.construct_atom_sets)


            * [maml.models.dl._keras_utils module](maml.models.dl.md#module-maml.models.dl._keras_utils)


                * [`deserialize_keras_activation()`](maml.models.dl.md#maml.models.dl._keras_utils.deserialize_keras_activation)


                * [`deserialize_keras_optimizer()`](maml.models.dl.md#maml.models.dl._keras_utils.deserialize_keras_optimizer)


            * [maml.models.dl._layers module](maml.models.dl.md#module-maml.models.dl._layers)


                * [`WeightedAverageLayer`](maml.models.dl.md#maml.models.dl._layers.WeightedAverageLayer)


                    * [`WeightedAverageLayer.build()`](maml.models.dl.md#maml.models.dl._layers.WeightedAverageLayer.build)


                    * [`WeightedAverageLayer.call()`](maml.models.dl.md#maml.models.dl._layers.WeightedAverageLayer.call)


                    * [`WeightedAverageLayer.compute_output_shape()`](maml.models.dl.md#maml.models.dl._layers.WeightedAverageLayer.compute_output_shape)


                    * [`WeightedAverageLayer.get_config()`](maml.models.dl.md#maml.models.dl._layers.WeightedAverageLayer.get_config)


                    * [`WeightedAverageLayer.reduce_sum()`](maml.models.dl.md#maml.models.dl._layers.WeightedAverageLayer.reduce_sum)


                * [`WeightedSet2Set`](maml.models.dl.md#maml.models.dl._layers.WeightedSet2Set)


                    * [`WeightedSet2Set._lstm()`](maml.models.dl.md#maml.models.dl._layers.WeightedSet2Set._lstm)


                    * [`WeightedSet2Set.build()`](maml.models.dl.md#maml.models.dl._layers.WeightedSet2Set.build)


                    * [`WeightedSet2Set.call()`](maml.models.dl.md#maml.models.dl._layers.WeightedSet2Set.call)


                    * [`WeightedSet2Set.compute_output_shape()`](maml.models.dl.md#maml.models.dl._layers.WeightedSet2Set.compute_output_shape)


                    * [`WeightedSet2Set.get_config()`](maml.models.dl.md#maml.models.dl._layers.WeightedSet2Set.get_config)


            * [maml.models.dl._mlp module](maml.models.dl.md#module-maml.models.dl._mlp)


                * [`MLP`](maml.models.dl.md#maml.models.dl._mlp.MLP)


                * [`construct_mlp()`](maml.models.dl.md#maml.models.dl._mlp.construct_mlp)


* [maml.sampling package](maml.sampling.md)




    * [maml.sampling.clustering module](maml.sampling.md#module-maml.sampling.clustering)


        * [`BirchClustering`](maml.sampling.md#maml.sampling.clustering.BirchClustering)


            * [`BirchClustering._sklearn_auto_wrap_output_keys`](maml.sampling.md#maml.sampling.clustering.BirchClustering._sklearn_auto_wrap_output_keys)


            * [`BirchClustering.fit()`](maml.sampling.md#maml.sampling.clustering.BirchClustering.fit)


            * [`BirchClustering.transform()`](maml.sampling.md#maml.sampling.clustering.BirchClustering.transform)


    * [maml.sampling.direct module](maml.sampling.md#maml-sampling-direct-module)


    * [maml.sampling.pca module](maml.sampling.md#module-maml.sampling.pca)


        * [`PrincipalComponentAnalysis`](maml.sampling.md#maml.sampling.pca.PrincipalComponentAnalysis)


            * [`PrincipalComponentAnalysis._sklearn_auto_wrap_output_keys`](maml.sampling.md#maml.sampling.pca.PrincipalComponentAnalysis._sklearn_auto_wrap_output_keys)


            * [`PrincipalComponentAnalysis.fit()`](maml.sampling.md#maml.sampling.pca.PrincipalComponentAnalysis.fit)


            * [`PrincipalComponentAnalysis.transform()`](maml.sampling.md#maml.sampling.pca.PrincipalComponentAnalysis.transform)


    * [maml.sampling.stratified_sampling module](maml.sampling.md#module-maml.sampling.stratified_sampling)


        * [`SelectKFromClusters`](maml.sampling.md#maml.sampling.stratified_sampling.SelectKFromClusters)


            * [`SelectKFromClusters._sklearn_auto_wrap_output_keys`](maml.sampling.md#maml.sampling.stratified_sampling.SelectKFromClusters._sklearn_auto_wrap_output_keys)


            * [`SelectKFromClusters.fit()`](maml.sampling.md#maml.sampling.stratified_sampling.SelectKFromClusters.fit)


            * [`SelectKFromClusters.transform()`](maml.sampling.md#maml.sampling.stratified_sampling.SelectKFromClusters.transform)


* [maml.utils package](maml.utils.md)


    * [`ConstantValue`](maml.utils.md#maml.utils.ConstantValue)


        * [`ConstantValue.get_value()`](maml.utils.md#maml.utils.ConstantValue.get_value)


    * [`DataSplitter`](maml.utils.md#maml.utils.DataSplitter)


        * [`DataSplitter.split()`](maml.utils.md#maml.utils.DataSplitter.split)


    * [`DummyScaler`](maml.utils.md#maml.utils.DummyScaler)


        * [`DummyScaler.from_training_data()`](maml.utils.md#maml.utils.DummyScaler.from_training_data)


        * [`DummyScaler.inverse_transform()`](maml.utils.md#maml.utils.DummyScaler.inverse_transform)


        * [`DummyScaler.transform()`](maml.utils.md#maml.utils.DummyScaler.transform)


    * [`LinearProfile`](maml.utils.md#maml.utils.LinearProfile)


        * [`LinearProfile.get_value()`](maml.utils.md#maml.utils.LinearProfile.get_value)


    * [`MultiScratchDir`](maml.utils.md#maml.utils.MultiScratchDir)


        * [`MultiScratchDir.SCR_LINK`](maml.utils.md#maml.utils.MultiScratchDir.SCR_LINK)


    * [`Scaler`](maml.utils.md#maml.utils.Scaler)


        * [`Scaler.inverse_transform()`](maml.utils.md#maml.utils.Scaler.inverse_transform)


        * [`Scaler.transform()`](maml.utils.md#maml.utils.Scaler.transform)


    * [`ShuffleSplitter`](maml.utils.md#maml.utils.ShuffleSplitter)


        * [`ShuffleSplitter.split()`](maml.utils.md#maml.utils.ShuffleSplitter.split)


    * [`StandardScaler`](maml.utils.md#maml.utils.StandardScaler)


        * [`StandardScaler.from_training_data()`](maml.utils.md#maml.utils.StandardScaler.from_training_data)


        * [`StandardScaler.inverse_transform()`](maml.utils.md#maml.utils.StandardScaler.inverse_transform)


        * [`StandardScaler.transform()`](maml.utils.md#maml.utils.StandardScaler.transform)


    * [`Stats`](maml.utils.md#maml.utils.Stats)


        * [`Stats.allowed_stats`](maml.utils.md#maml.utils.Stats.allowed_stats)


        * [`Stats.average()`](maml.utils.md#maml.utils.Stats.average)


        * [`Stats.geometric_mean()`](maml.utils.md#maml.utils.Stats.geometric_mean)


        * [`Stats.harmonic_mean()`](maml.utils.md#maml.utils.Stats.harmonic_mean)


        * [`Stats.inverse_mean()`](maml.utils.md#maml.utils.Stats.inverse_mean)


        * [`Stats.kurtosis()`](maml.utils.md#maml.utils.Stats.kurtosis)


        * [`Stats.max()`](maml.utils.md#maml.utils.Stats.max)


        * [`Stats.mean()`](maml.utils.md#maml.utils.Stats.mean)


        * [`Stats.mean_absolute_deviation()`](maml.utils.md#maml.utils.Stats.mean_absolute_deviation)


        * [`Stats.mean_absolute_error()`](maml.utils.md#maml.utils.Stats.mean_absolute_error)


        * [`Stats.min()`](maml.utils.md#maml.utils.Stats.min)


        * [`Stats.mode()`](maml.utils.md#maml.utils.Stats.mode)


        * [`Stats.moment()`](maml.utils.md#maml.utils.Stats.moment)


        * [`Stats.power_mean()`](maml.utils.md#maml.utils.Stats.power_mean)


        * [`Stats.range()`](maml.utils.md#maml.utils.Stats.range)


        * [`Stats.shifted_geometric_mean()`](maml.utils.md#maml.utils.Stats.shifted_geometric_mean)


        * [`Stats.skewness()`](maml.utils.md#maml.utils.Stats.skewness)


        * [`Stats.std()`](maml.utils.md#maml.utils.Stats.std)


    * [`ValueProfile`](maml.utils.md#maml.utils.ValueProfile)


        * [`ValueProfile.get_value()`](maml.utils.md#maml.utils.ValueProfile.get_value)


        * [`ValueProfile.get_value()`](maml.utils.md#id0)


        * [`ValueProfile.increment_step()`](maml.utils.md#maml.utils.ValueProfile.increment_step)


    * [`check_structures_forces_stresses()`](maml.utils.md#maml.utils.check_structures_forces_stresses)


    * [`convert_docs()`](maml.utils.md#maml.utils.convert_docs)


    * [`cwt()`](maml.utils.md#maml.utils.cwt)


    * [`feature_dim_from_test_system()`](maml.utils.md#maml.utils.feature_dim_from_test_system)


    * [`fft_magnitude()`](maml.utils.md#maml.utils.fft_magnitude)


    * [`get_describer_dummy_obj()`](maml.utils.md#maml.utils.get_describer_dummy_obj)


    * [`get_full_args()`](maml.utils.md#maml.utils.get_full_args)


    * [`get_full_stats_and_funcs()`](maml.utils.md#maml.utils.get_full_stats_and_funcs)


    * [`get_lammps_lattice_and_rotation()`](maml.utils.md#maml.utils.get_lammps_lattice_and_rotation)


    * [`get_sp_method()`](maml.utils.md#maml.utils.get_sp_method)


    * [`njit()`](maml.utils.md#maml.utils.njit)


    * [`pool_from()`](maml.utils.md#maml.utils.pool_from)


    * [`spectrogram()`](maml.utils.md#maml.utils.spectrogram)


    * [`stats_list_conversion()`](maml.utils.md#maml.utils.stats_list_conversion)


    * [`stress_format_change()`](maml.utils.md#maml.utils.stress_format_change)


    * [`stress_list_to_matrix()`](maml.utils.md#maml.utils.stress_list_to_matrix)


    * [`stress_matrix_to_list()`](maml.utils.md#maml.utils.stress_matrix_to_list)


    * [`to_array()`](maml.utils.md#maml.utils.to_array)


    * [`to_composition()`](maml.utils.md#maml.utils.to_composition)


    * [`write_data_from_structure()`](maml.utils.md#maml.utils.write_data_from_structure)


    * [`wvd()`](maml.utils.md#maml.utils.wvd)




    * [maml.utils._data_conversion module](maml.utils.md#module-maml.utils._data_conversion)


        * [`convert_docs()`](maml.utils.md#maml.utils._data_conversion.convert_docs)


        * [`doc_from()`](maml.utils.md#maml.utils._data_conversion.doc_from)


        * [`pool_from()`](maml.utils.md#maml.utils._data_conversion.pool_from)


        * [`to_array()`](maml.utils.md#maml.utils._data_conversion.to_array)


    * [maml.utils._data_split module](maml.utils.md#module-maml.utils._data_split)


        * [`DataSplitter`](maml.utils.md#maml.utils._data_split.DataSplitter)


            * [`DataSplitter.split()`](maml.utils.md#maml.utils._data_split.DataSplitter.split)


        * [`ShuffleSplitter`](maml.utils.md#maml.utils._data_split.ShuffleSplitter)


            * [`ShuffleSplitter.split()`](maml.utils.md#maml.utils._data_split.ShuffleSplitter.split)


    * [maml.utils._dummy module](maml.utils.md#module-maml.utils._dummy)


        * [`feature_dim_from_test_system()`](maml.utils.md#maml.utils._dummy.feature_dim_from_test_system)


        * [`get_describer_dummy_obj()`](maml.utils.md#maml.utils._dummy.get_describer_dummy_obj)


    * [maml.utils._inspect module](maml.utils.md#module-maml.utils._inspect)


        * [`get_full_args()`](maml.utils.md#maml.utils._inspect.get_full_args)


        * [`get_param_types()`](maml.utils.md#maml.utils._inspect.get_param_types)


    * [maml.utils._jit module](maml.utils.md#module-maml.utils._jit)


        * [`njit()`](maml.utils.md#maml.utils._jit.njit)


    * [maml.utils._lammps module](maml.utils.md#module-maml.utils._lammps)


        * [`_get_atomic_mass()`](maml.utils.md#maml.utils._lammps._get_atomic_mass)


        * [`_get_charge()`](maml.utils.md#maml.utils._lammps._get_charge)


        * [`check_structures_forces_stresses()`](maml.utils.md#maml.utils._lammps.check_structures_forces_stresses)


        * [`get_lammps_lattice_and_rotation()`](maml.utils.md#maml.utils._lammps.get_lammps_lattice_and_rotation)


        * [`stress_format_change()`](maml.utils.md#maml.utils._lammps.stress_format_change)


        * [`stress_list_to_matrix()`](maml.utils.md#maml.utils._lammps.stress_list_to_matrix)


        * [`stress_matrix_to_list()`](maml.utils.md#maml.utils._lammps.stress_matrix_to_list)


        * [`write_data_from_structure()`](maml.utils.md#maml.utils._lammps.write_data_from_structure)


    * [maml.utils._material module](maml.utils.md#module-maml.utils._material)


        * [`to_composition()`](maml.utils.md#maml.utils._material.to_composition)


    * [maml.utils._preprocessing module](maml.utils.md#module-maml.utils._preprocessing)


        * [`DummyScaler`](maml.utils.md#maml.utils._preprocessing.DummyScaler)


            * [`DummyScaler.from_training_data()`](maml.utils.md#maml.utils._preprocessing.DummyScaler.from_training_data)


            * [`DummyScaler.inverse_transform()`](maml.utils.md#maml.utils._preprocessing.DummyScaler.inverse_transform)


            * [`DummyScaler.transform()`](maml.utils.md#maml.utils._preprocessing.DummyScaler.transform)


        * [`Scaler`](maml.utils.md#maml.utils._preprocessing.Scaler)


            * [`Scaler.inverse_transform()`](maml.utils.md#maml.utils._preprocessing.Scaler.inverse_transform)


            * [`Scaler.transform()`](maml.utils.md#maml.utils._preprocessing.Scaler.transform)


        * [`StandardScaler`](maml.utils.md#maml.utils._preprocessing.StandardScaler)


            * [`StandardScaler.from_training_data()`](maml.utils.md#maml.utils._preprocessing.StandardScaler.from_training_data)


            * [`StandardScaler.inverse_transform()`](maml.utils.md#maml.utils._preprocessing.StandardScaler.inverse_transform)


            * [`StandardScaler.transform()`](maml.utils.md#maml.utils._preprocessing.StandardScaler.transform)


    * [maml.utils._signal_processing module](maml.utils.md#module-maml.utils._signal_processing)


        * [`cwt()`](maml.utils.md#maml.utils._signal_processing.cwt)


        * [`fft_magnitude()`](maml.utils.md#maml.utils._signal_processing.fft_magnitude)


        * [`get_sp_method()`](maml.utils.md#maml.utils._signal_processing.get_sp_method)


        * [`spectrogram()`](maml.utils.md#maml.utils._signal_processing.spectrogram)


        * [`wvd()`](maml.utils.md#maml.utils._signal_processing.wvd)


    * [maml.utils._stats module](maml.utils.md#module-maml.utils._stats)


        * [`Stats`](maml.utils.md#maml.utils._stats.Stats)


            * [`Stats.allowed_stats`](maml.utils.md#maml.utils._stats.Stats.allowed_stats)


            * [`Stats.average()`](maml.utils.md#maml.utils._stats.Stats.average)


            * [`Stats.geometric_mean()`](maml.utils.md#maml.utils._stats.Stats.geometric_mean)


            * [`Stats.harmonic_mean()`](maml.utils.md#maml.utils._stats.Stats.harmonic_mean)


            * [`Stats.inverse_mean()`](maml.utils.md#maml.utils._stats.Stats.inverse_mean)


            * [`Stats.kurtosis()`](maml.utils.md#maml.utils._stats.Stats.kurtosis)


            * [`Stats.max()`](maml.utils.md#maml.utils._stats.Stats.max)


            * [`Stats.mean()`](maml.utils.md#maml.utils._stats.Stats.mean)


            * [`Stats.mean_absolute_deviation()`](maml.utils.md#maml.utils._stats.Stats.mean_absolute_deviation)


            * [`Stats.mean_absolute_error()`](maml.utils.md#maml.utils._stats.Stats.mean_absolute_error)


            * [`Stats.min()`](maml.utils.md#maml.utils._stats.Stats.min)


            * [`Stats.mode()`](maml.utils.md#maml.utils._stats.Stats.mode)


            * [`Stats.moment()`](maml.utils.md#maml.utils._stats.Stats.moment)


            * [`Stats.power_mean()`](maml.utils.md#maml.utils._stats.Stats.power_mean)


            * [`Stats.range()`](maml.utils.md#maml.utils._stats.Stats.range)


            * [`Stats.shifted_geometric_mean()`](maml.utils.md#maml.utils._stats.Stats.shifted_geometric_mean)


            * [`Stats.skewness()`](maml.utils.md#maml.utils._stats.Stats.skewness)


            * [`Stats.std()`](maml.utils.md#maml.utils._stats.Stats.std)


        * [`_add_allowed_stats()`](maml.utils.md#maml.utils._stats._add_allowed_stats)


        * [`_convert_a_or_b()`](maml.utils.md#maml.utils._stats._convert_a_or_b)


        * [`_moment_symbol_conversion()`](maml.utils.md#maml.utils._stats._moment_symbol_conversion)


        * [`_root_moment()`](maml.utils.md#maml.utils._stats._root_moment)


        * [`get_full_stats_and_funcs()`](maml.utils.md#maml.utils._stats.get_full_stats_and_funcs)


        * [`stats_list_conversion()`](maml.utils.md#maml.utils._stats.stats_list_conversion)


    * [maml.utils._tempfile module](maml.utils.md#module-maml.utils._tempfile)


        * [`MultiScratchDir`](maml.utils.md#maml.utils._tempfile.MultiScratchDir)


            * [`MultiScratchDir.SCR_LINK`](maml.utils.md#maml.utils._tempfile.MultiScratchDir.SCR_LINK)


            * [`MultiScratchDir.tempdirs`](maml.utils.md#maml.utils._tempfile.MultiScratchDir.tempdirs)


        * [`_copy_r_with_suffix()`](maml.utils.md#maml.utils._tempfile._copy_r_with_suffix)


    * [maml.utils._typing module](maml.utils.md#module-maml.utils._typing)


    * [maml.utils._value_profile module](maml.utils.md#module-maml.utils._value_profile)


        * [`ConstantValue`](maml.utils.md#maml.utils._value_profile.ConstantValue)


            * [`ConstantValue.get_value()`](maml.utils.md#maml.utils._value_profile.ConstantValue.get_value)


        * [`LinearProfile`](maml.utils.md#maml.utils._value_profile.LinearProfile)


            * [`LinearProfile.get_value()`](maml.utils.md#maml.utils._value_profile.LinearProfile.get_value)


        * [`ValueProfile`](maml.utils.md#maml.utils._value_profile.ValueProfile)


            * [`ValueProfile.get_value()`](maml.utils.md#maml.utils._value_profile.ValueProfile.get_value)


            * [`ValueProfile.get_value()`](maml.utils.md#id1)


            * [`ValueProfile.increment_step()`](maml.utils.md#maml.utils._value_profile.ValueProfile.increment_step)