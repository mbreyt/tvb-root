# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Adapter that uses the traits module to generate interfaces for BalloonModel Analyzer.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import uuid

import numpy
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.datatypes.h5.time_series_h5 import TimeSeriesRegionH5
from tvb.analyzers.fmri_balloon import BalloonModel
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.db import prepare_array_shape_meta
from tvb.core.neotraits.forms import TraitDataTypeSelectField, FloatField, StrField, BoolField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.time_series import TimeSeries, TimeSeriesRegion
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Range, Float
import tvb.simulator.integrators as integrators_module


class BalloonModelAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeriesRegion,
        label="Time Series",
        required=True,
        doc="""The timeseries that represents the input neural activity"""
    )

    dt = Float(
        label=":math:`dt`",
        default=0.002,
        required=True,
        doc="""The integration time step size for the balloon model (s).
            If none is provided, by default, the TimeSeries sample period is used."""
    )

    tau_s = Float(
        label=r":math:`\tau_s`",
        default=1.54,
        required=True,
        doc="""Balloon model parameter. Time of signal decay (s)""")

    tau_f = Float(
        label=r":math:`\tau_f`",
        default=1.44,
        required=True,
        doc=""" Balloon model parameter. Time of flow-dependent elimination or
            feedback regulation (s). The average  time blood take to traverse the
            venous compartment. It is the  ratio of resting blood volume (V0) to
            resting blood flow (F0).""")

    neural_input_transformation = Attr(
        field_type=str,
        label="Neural input transformation",
        choices=("none", "abs_diff", "sum"),
        default="none",
        doc=""" This represents the operation to perform on the state-variable(s) of
            the model used to generate the input TimeSeries. ``none`` takes the
            first state-variable as neural input; `` abs_diff`` is the absolute
            value of the derivative (first order difference) of the first state variable; 
            ``sum``: sum all the state-variables of the input TimeSeries."""
    )

    bold_model = Attr(
        field_type=str,
        label="Select BOLD model equations",
        choices=("linear", "nonlinear"),
        default="nonlinear",
        doc="""Select the set of equations for the BOLD model."""
    )

    RBM = Attr(
        field_type=bool,
        label="Revised BOLD Model",
        default=True,
        required=True,
        doc="""Select classical vs revised BOLD model (CBM or RBM).
            Coefficients  k1, k2 and k3 will be derived accordingly."""
    )

    integrator = Attr(
        field_type=integrators_module.Integrator,
        label="Integration scheme",
        default=integrators_module.HeunDeterministic(),
        required=True,
        doc=""" A tvb.simulator.Integrator object which is
        an integration scheme with supporting attributes such as 
        integration step size and noise specification for stochastic 
        methods. It is used to compute the time courses of the balloon model state 
        variables.""")

    tau_o = Float(
        label=r":math:`\tau_o`",
        default=0.98,
        required=True,
        doc="""
        Balloon model parameter. Haemodynamic transit time (s). The average
        time blood take to traverse the venous compartment. It is the  ratio
        of resting blood volume (V0) to resting blood flow (F0).""")

    alpha = Float(
        label=r":math:`\tau_f`",
        default=0.32,
        required=True,
        doc="""Balloon model parameter. Stiffness parameter. Grubb's exponent.""")

    TE = Float(
        label=":math:`TE`",
        default=0.04,
        required=True,
        doc="""BOLD parameter. Echo Time""")

    V0 = Float(
        label=":math:`V_0`",
        default=4.0,
        required=True,
        doc="""BOLD parameter. Resting blood volume fraction.""")

    E0 = Float(
        label=":math:`E_0`",
        default=0.4,
        required=True,
        doc="""BOLD parameter. Resting oxygen extraction fraction.""")

    epsilon = NArray(
        label=":math:`\epsilon`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.5, hi=2.0, step=0.25),
        required=True,
        doc=""" BOLD parameter. Ratio of intra- and extravascular signals. In principle  this
        parameter could be derived from empirical data and spatialized.""")

    nu_0 = Float(
        label=r":math:`\nu_0`",
        default=40.3,
        required=True,
        doc="""BOLD parameter. Frequency offset at the outer surface of magnetized vessels (Hz).""")

    r_0 = Float(
        label=":math:`r_0`",
        default=25.,
        required=True,
        doc=""" BOLD parameter. Slope r0 of intravascular relaxation rate (Hz). Only used for
        ``revised`` coefficients. """)


class BalloonModelAdapterForm(ABCAdapterForm):

    def __init__(self, project_id=None):
        super(BalloonModelAdapterForm, self).__init__(project_id)
        self.time_series = TraitDataTypeSelectField(BalloonModelAdapterModel.time_series, self.project_id,
                                                    name=self.get_input_name(), conditions=self.get_filters(),
                                                    has_all_option=True)
        self.dt = FloatField(BalloonModelAdapterModel.dt, self.project_id)
        self.tau_s = FloatField(BalloonModelAdapterModel.tau_s, self.project_id)
        self.tau_f = FloatField(BalloonModelAdapterModel.tau_f, self.project_id)
        self.neural_input_transformation = StrField(BalloonModelAdapterModel.neural_input_transformation,
                                                       self.project_id)
        self.bold_model = StrField(BalloonModelAdapterModel.bold_model, self.project_id)
        self.RBM = BoolField(BalloonModelAdapterModel.RBM, self.project_id)

    @staticmethod
    def get_view_model():
        return BalloonModelAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesRegionIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return 'time_series'


class BalloonModelAdapter(ABCAdapter):
    """
    TVB adapter for calling the BalloonModel algorithm.
    """

    _ui_name = "Balloon Model "
    _ui_description = "Compute BOLD signals for a TimeSeries input DataType."
    _ui_subsection = "balloon"

    def get_form_class(self):
        return BalloonModelAdapterForm

    def get_output(self):
        return [TimeSeriesRegionIndex]

    def configure(self, view_model):
        # type: (BalloonModelAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage. Also
        create the algorithm instance.
        """
        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)

        self.log.debug("time_series shape is %s" % str(self.input_shape))
        # -------------------- Fill Algorithm for Analysis -------------------##

        if view_model.dt is None:
            view_model.dt = self.input_time_series_index.sample_period / 1000.
        algorithm = BalloonModel(view_model.time_series, view_model.dt, view_model.integrator,
                                 view_model.bold_model,
                                 view_model.RBM, view_model.neural_input_transformation, view_model.tau_s,
                                 view_model.tau_f, view_model.tau_o, view_model.alpha, view_model.TE,
                                 view_model.V0,
                                 view_model.E0, view_model.epsilon, view_model.nu_0, view_model.r_0)
        self.algorithm = algorithm

    def get_required_memory_size(self, view_model):
        # type: (BalloonModelAdapterModel) -> int
        """
        Return the required memory to run this algorithm.
        """
        used_shape = self.input_shape
        input_size = numpy.prod(used_shape) * 8.0
        output_size = self.result_size(used_shape)
        return input_size + output_size

    def get_required_disk_size(self, view_model):
        # type: (BalloonModelAdapterModel) -> int
        """
        Returns the required disk size to be able to run the adapter.(in kB)
        """
        used_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3])
        return self.array_size2kb(self.result_size(used_shape))

    def launch(self, view_model):
        # type: (BalloonModelAdapterModel) -> [TimeSeriesRegionIndex]
        """
        Launch algorithm and build results.

        :param time_series: the input time-series used as neural activation in the Balloon Model
        :returns: the simulated BOLD signal
        :rtype: `TimeSeries`
        """
        input_time_series_h5 = h5.h5_file_for_index(self.input_time_series_index)
        time_line = input_time_series_h5.read_time_page(0, self.input_shape[0])

        bold_signal_index = TimeSeriesRegionIndex()
        bold_signal_h5_path = h5.path_for(self.storage_path, TimeSeriesRegionH5, bold_signal_index.gid)
        bold_signal_h5 = TimeSeriesRegionH5(bold_signal_h5_path)
        bold_signal_h5.gid.store(uuid.UUID(bold_signal_index.gid))
        self._fill_result_h5(bold_signal_h5, input_time_series_h5)

        # ---------- Iterate over slices and compose final result ------------##

        node_slice = [slice(self.input_shape[0]), slice(self.input_shape[1]), None, slice(self.input_shape[3])]
        small_ts = TimeSeries()
        small_ts.sample_period = self.input_time_series_index.sample_period
        small_ts.sample_period_unit = self.input_time_series_index.sample_period_unit
        small_ts.time = time_line

        for node in range(self.input_shape[2]):
            node_slice[2] = slice(node, node + 1)
            small_ts.data = input_time_series_h5.read_data_slice(tuple(node_slice))
            self.algorithm.time_series = small_ts
            partial_bold = self.algorithm.calculate_simulated_bold_signal()
            bold_signal_h5.write_data_slice_on_grow_dimension(partial_bold.data, grow_dimension=2)

        bold_signal_h5.write_time_slice(time_line)
        bold_signal_shape = bold_signal_h5.data.shape
        bold_signal_h5.nr_dimensions.store(len(bold_signal_shape))
        bold_signal_h5.close()
        input_time_series_h5.close()

        self._fill_result_index(bold_signal_index, bold_signal_shape)
        return bold_signal_index

    def _fill_result_index(self, result_index, result_signal_shape):
        result_index.time_series_type = TimeSeriesRegion.__name__
        result_index.data_ndim = len(result_signal_shape)
        result_index.data_length_1d, result_index.data_length_2d, \
        result_index.data_length_3d, result_index.data_length_4d = prepare_array_shape_meta(result_signal_shape)

        result_index.fk_connectivity_gid = self.input_time_series_index.fk_connectivity_gid
        result_index.fk_region_mapping_gid = self.input_time_series_index.fk_region_mapping_gid
        result_index.fk_region_mapping_volume_gid = self.input_time_series_index.fk_region_mapping_volume_gid

        result_index.sample_period = self.input_time_series_index.sample_period
        result_index.sample_period_unit = self.input_time_series_index.sample_period_unit
        result_index.sample_rate = self.input_time_series_index.sample_rate
        result_index.labels_ordering = self.input_time_series_index.labels_ordering
        result_index.labels_dimensions = self.input_time_series_index.labels_dimensions
        result_index.has_volume_mapping = self.input_time_series_index.has_volume_mapping
        result_index.has_surface_mapping = self.input_time_series_index.has_surface_mapping
        result_index.title = self.input_time_series_index.title

    def _fill_result_h5(self, result_h5, input_h5):
        result_h5.sample_period.store(self.input_time_series_index.sample_period)
        result_h5.sample_period_unit.store(self.input_time_series_index.sample_period_unit)
        result_h5.sample_rate.store(input_h5.sample_rate.load())
        result_h5.start_time.store(input_h5.start_time.load())
        result_h5.labels_ordering.store(input_h5.labels_ordering.load())
        result_h5.labels_dimensions.store(input_h5.labels_dimensions.load())
        result_h5.connectivity.store(input_h5.connectivity.load())
        result_h5.region_mapping_volume.store(input_h5.region_mapping_volume.load())
        result_h5.region_mapping.store(input_h5.region_mapping.load())
        result_h5.title.store(input_h5.title.load())

    @staticmethod
    def result_shape(input_shape):
        """Returns the shape of the main result of fmri balloon ..."""
        result_shape = (input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3])
        return result_shape

    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = numpy.sum(list(map(numpy.prod, self.result_shape(input_shape)))) * 8.0  # Bytes
        return result_size
